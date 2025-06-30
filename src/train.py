import os
import re
import sys
import gin
import time
import contextlib
import torch
import torch.nn as nn
import math
from typing import Type
from loguru import logger
from adamw_bf16 import AdamWBF16
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torcheval.metrics import BinaryAUROC

from data import SeqRecDataset
from model import ModelArgs, Transformer
from common import latest_checkpoint


class SpeedTracker(object):
    def __init__(self, buffer_size=10000):
        self.times = []
        self.buffer_size = buffer_size

    def update(self):
        self.times.append(time.time())
        if len(self.times) > 2 * self.buffer_size:
            self.times = self.times[-self.buffer_size:]

    def get_speed(self, steps=100):        
        if len(self.times) < 2:
            return 0
        steps = min(len(self.times)-1, steps)
        return steps / (self.times[-1]-self.times[-steps-1])

def collate_fn(batch):
    # filter None samples
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)

@gin.configurable
@torch.no_grad()
def eval(
        model:torch.nn.Module,
        summary_writer:SummaryWriter,
        data_path:str=None,
        batch_size:int=4096,
        num_data_workers:int=2,
        prefetch_factor:int=32,
        log_steps:int=100,
        global_step:int=0,
        max_steps:int=200,
        drop_last:bool=True
    ):
    if data_path is None:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    dataset = SeqRecDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers, pin_memory=True, drop_last=drop_last, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
    logger.info(f"eval dataset: {len(dataset)} eval dataloader: {len(dataloader)}")
    model.eval()
    rank_metric = BinaryAUROC()
    rank_loss_sum = 0
    steps = 0
    total_steps = min(max_steps, len(dataloader)) if max_steps > 0 else len(dataloader)
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)
        ret = model(**batch)
        rank_predicts = ret["rank_outputs"]
        rank_loss = ret["rank_loss"]
        rank_labels = batch["click_label"]
        steps += 1
        if world_size > 1:
            rank_predicts_gather_list = [torch.zeros_like(rank_predicts) for _ in range(world_size)]
            rank_labels_gather_list = [torch.zeros_like(rank_labels) for _ in range(world_size)]
            dist.all_gather(rank_predicts_gather_list, rank_predicts)
            dist.all_gather(rank_labels_gather_list, rank_labels)
            rank_predicts = torch.cat(rank_predicts_gather_list, dim=0)
            rank_labels = torch.cat(rank_labels_gather_list, dim=0)
            dist.all_reduce(rank_loss, op=dist.ReduceOp.SUM)
            rank_loss /= world_size
        rank_loss_sum += rank_loss
        rank_mask = (rank_labels == 0) | (rank_labels == 1)
        rank_metric.update(rank_predicts[rank_mask], rank_labels[rank_mask])
        if local_rank == 0 and steps % log_steps == 0:
            rank_auc = rank_metric.compute()
            rank_loss = rank_loss_sum/steps
            logger.info(f"eval steps: {steps}/{total_steps}, rank_loss: {rank_loss:.4f} rank_auc: {rank_auc:.4f}")
        if max_steps > 0 and steps >= max_steps:
            break
    del dataloader
    rank_auc = rank_metric.compute()
    rank_loss = rank_loss_sum/steps
    if local_rank == 0:
        logger.info(f"eval metrics: global_step: {global_step} rank_loss: {rank_loss:.4f} rank_auc: {rank_auc:.4f}")
    if rank == 0:
        summary_writer.add_scalar("eval/rank_loss", rank_loss.to(torch.float32), global_step)
        summary_writer.add_scalar("eval/rank_auc", rank_auc.to(torch.float32), global_step)
    model.train()
    return rank_auc.to(torch.float32).item()


@gin.configurable
def train(
        epoch:int=1,
        batch_size:int=4096,
        data_path:str=None,
        num_data_workers:int=2,
        prefetch_factor:int=32,
        ckpt_root_dir:str=None,
        ckpt_name:str=None,
        learning_rate:float=5e-4,
        min_learning_rate:float=None,
        warmup_steps=0,
        adam_betas:tuple=(0.9, 0.98),
        weight_decay:float=0,
        max_grads_norm=0,
        shuffle=True,
        restore=True,
        use_bf16:bool=False,
        log_steps:int=100,
        model_cls:Type=Transformer,
        frozen_params:str=None,
        load_ckpt:str=None,
        load_params:str=".*",
        random_seed=17,
        eval_steps=1000,
        grad_accumulate_steps=1
        ):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    checkpoint_dir = os.path.join(ckpt_root_dir, ckpt_name)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        summary_writer = SummaryWriter(checkpoint_dir)
    else:
        summary_writer = None
    logger.add(open(os.path.join(checkpoint_dir, f"log-{rank}.txt"), "w"))
    logger.info(f"WORLD_SIZE: {world_size} RANK: {rank} LOCAL_RANK: {local_rank}")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    logger.info("checkpoint_dir: {}".format(checkpoint_dir))
    # init dist
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # build datasets
    dataset = SeqRecDataset(data_path)
    if rank == 0:
        dataset.save_tokenizers(checkpoint_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_data_workers, pin_memory=True, drop_last=True, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
    logger.info(f"train dataset: {len(dataset)} train dataloader: {len(dataloader)}")
    # build model args
    model_args = ModelArgs()
    model_args.item_vocab_size=dataset.item_tokenizer.get_vocab_size()
    model_args.cate_vocab_size=dataset.cate_tokenizer.get_vocab_size()
    # build model
    model_: nn.Module = model_cls(model_args, seed=random_seed)
    model_ = model_.to(device)
    # count params
    n_sparse_params = n_dense_params = 0
    for name, param in model_.named_parameters():
        if name.startswith("item_embeddings"):
            n_sparse_params += param.numel()
        else:
            n_dense_params += param.numel()
    logger.info(f"#Sparse: {n_sparse_params} #Dense: {n_dense_params}")

    if world_size > 1:
        model = DDP(model_, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = model_

    if use_bf16:
        model_ = model_.to(torch.bfloat16)
        logger.info(f"Convert model to bfloat16.")

    # build optimizer
    opt_cls = AdamWBF16 if use_bf16 else torch.optim.AdamW
    opt = opt_cls(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=weight_decay,
    )

    # set frozen params
    if frozen_params:
        logger.info(f"frozen_param: {frozen_params}")
        pattern = re.compile(frozen_params)
        for name, param in model_.named_parameters():
            if pattern.match(name):
                param.requires_grad = False
                logger.info(f"Frozen param: {name}")

    # prepare training
    global_step = 0
    start_epoch = 0
    batch_index = 0
    epoch_steps = len(dataloader) // grad_accumulate_steps
    total_steps = epoch_steps * epoch
    rank_metric = BinaryAUROC()

    restored = False
    if restore:
        latest_ckpt = latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            ckpt = torch.load(open(latest_ckpt, 'rb'))
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt["steps"]
            start_epoch = ckpt["epoch"]
            del ckpt
            restored = True

    if load_ckpt and not restored:
        if os.path.isdir(load_ckpt):
            load_ckpt = latest_checkpoint(load_ckpt)
        ckpt = torch.load(open(load_ckpt, 'rb'))
        ckpt["model_state_dict"] = {re.sub('^module.', '', k): v for k, v in ckpt["model_state_dict"].items()} # remove wrapper
        pattern = re.compile(load_params)
        for key in list(ckpt["model_state_dict"].keys()):
            if not pattern.match(key):
                logger.info(f"Excluded key during loading: {key}.")
                ckpt["model_state_dict"].pop(key)
        ret = model_.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Missing keys: {ret.missing_keys}")
        logger.info(f"unexpected keys: {ret.unexpected_keys}")
        logger.info(f"Load from {load_ckpt} successfully.")
        del ckpt

    # set learning rate
    def schedule_lr(global_step, total_steps):
        peak_lr, min_lr = learning_rate, min_learning_rate
        assert min_lr is None or min_lr < peak_lr
        if warmup_steps > 0 and global_step < warmup_steps:
            # apply warm up
            lr = peak_lr * (global_step+1) / warmup_steps
        else:
            if min_lr is None:
                lr = peak_lr
            else:
                # apply cosine annealing
                cur_steps = min(global_step + 1 - warmup_steps, total_steps)
                max_steps = total_steps - warmup_steps
                lr = min_lr + (peak_lr - min_lr) * (1 + torch.cos(torch.tensor(cur_steps/max_steps*math.pi))) / 2.0
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        return lr

    speed_tracker = SpeedTracker()
    for ep in range(start_epoch, epoch):
        model.train()
        for batch in dataloader:
            batch_index += 1
            is_accumlate_step = batch_index % grad_accumulate_steps != 0
            for k in batch:
                batch[k] = batch[k].to(device)
            with model.no_sync() if is_accumlate_step and world_size > 1 else contextlib.nullcontext():
                ret = model(**batch)
                loss = ret['loss']
                item_ar_loss = ret['item_ar_loss']
                cate_ar_loss = ret['cate_ar_loss']
                rank_loss = ret['rank_loss']
                loss.backward()
                if is_accumlate_step:
                    continue
            if grad_accumulate_steps > 1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= grad_accumulate_steps
            if max_grads_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grads_norm)
            lr = schedule_lr(global_step, total_steps)
            opt.step()
            opt.zero_grad()
            global_step += 1
            speed_tracker.update()

            # update metrics
            rank_mask = (batch["click_label"] == 0) | (batch["click_label"] == 1)
            rank_metric.update(ret["rank_outputs"][rank_mask], batch["click_label"][rank_mask])

            if ((global_step-1) % log_steps == 0):
                if local_rank == 0:
                    rank_auc = rank_metric.compute() if torch.any(rank_mask) else torch.tensor(0.5)
                    logger.info(f"epoch: {ep+1}/{epoch} global_step: {global_step}/{total_steps}, loss: {loss:.4f}, item_ar_loss: {item_ar_loss:.4f}, cate_ar_loss: {cate_ar_loss:.4f}, rank_loss: {rank_loss:.4f}, rank_auc: {rank_auc:.4f}, speed: {speed_tracker.get_speed():.2f}steps/s, learning_rate: {lr:.6f}")
                if rank == 0:
                    rank_auc = rank_metric.compute() if torch.any(rank_mask) else torch.tensor(0.5)
                    summary_writer.add_scalar("train/loss", loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/item_ar_loss", item_ar_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/cate_ar_loss", cate_ar_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/rank_loss", rank_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/rank_auc", rank_auc.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/steps_per_second", speed_tracker.get_speed(), global_step)
                    summary_writer.add_scalar("train/learning_rate", lr, global_step)
                    for name, param in model_.named_parameters():
                        summary_writer.add_scalar(f"param_norm/{name}", param.norm().to(torch.float32), global_step)
                rank_metric.reset()
            
            # eval during epoch
            if eval_steps > 0 and global_step % eval_steps == 0:
                eval(model=model, summary_writer=summary_writer, global_step=global_step)
        # eval and save after epoch
        eval(model=model, summary_writer=summary_writer, global_step=global_step)
        if rank == 0:
            torch.save(
                    {
                        "epoch": ep+1,
                        "steps": global_step,
                        "model_state_dict": model_.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                    },
                    os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth"),
                )
        batch_index = 0 # reset

    if rank == 0:
        summary_writer.close()

    # destory dist
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="gin config file")
    args, _ = parser.parse_known_args()
    gin.parse_config_file(args.config)
    
    logger.remove()
    logger.add(sys.stdout, format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>] <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    train(ckpt_name=os.path.splitext(os.path.basename(args.config))[0])
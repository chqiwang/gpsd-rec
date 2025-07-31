from typing import Optional, Tuple
import torch
import gin
from torch import nn
import torch.nn.functional as F

from .args import ModelArgs
from .transformer import RankHead, FullARHead, SampledARHead

# some of the code is taken from https://github.com/meta-recsys/generative-recommenders

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len: int = max_seq_len
        self.weights = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        seq_len
    ) -> torch.Tensor:
        assert seq_len < self.max_seq_len
        t = F.pad(self.weights[self.max_seq_len - seq_len: self.max_seq_len + seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(1, 1, seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        return t[..., r:-r]


class HSTUBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.f1 = nn.Linear(
            args.dim, 4*args.dim
        )
        self.f2 = nn.Linear(
            args.dim, args.dim
        )
        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)
        self.pos_bias = RelativePositionalBias(args.max_seq_len)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xu, xq, xk, xv = F.silu(self.f1(x)).split(self.dim, dim=-1)
        xu = xu.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # [B,H,L,HD]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # [B,H,L,HD]
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # [B,H,L,HD]
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # [B,H,L,HD]

        att = F.silu(torch.matmul(xq, xk.transpose(2, 3)) + self.pos_bias(seqlen)) # [B,H,L,L]
        if mask is not None:
            att *= mask >= 0 # -inf means not to see
        y = torch.matmul(att, xv) * xu # [B,H,L,HD]
        y = y.transpose(1,2).contiguous().view(bsz, seqlen, -1) # [B,L,D]
        y = self.norm1(y)
        y = self.f2(y)
        y = self.norm2(self.dropout(y) + x)
        return y


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, add_pre_norm=True, add_post_norm=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.pre_norm = nn.Identity()
        self.post_norm = nn.Identity()
        if add_pre_norm:
            self.pre_norm = nn.LayerNorm(input_dim)
        if add_post_norm:
            self.post_norm = nn.LayerNorm(output_dim)

    def forward(self, input):
        return self.post_norm(self.linear(self.pre_norm(input)))


@gin.configurable
class HSTU(nn.Module):
    def __init__(self, params: ModelArgs, seed: int=0):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.item_embeddings = torch.nn.Embedding(params.item_vocab_size, params.embedding_dim or params.dim)
        self.cate_embeddings = torch.nn.Embedding(params.cate_vocab_size, params.embedding_dim or params.dim)
        self.segment_embeddings = torch.nn.Embedding(params.segment_vocab_size, params.embedding_dim or params.dim)

        if params.embedding_dim and params.dim != params.embedding_dim:
            self.pre_projector = ProjectionLayer(params.embedding_dim, params.dim)
            self.post_projector = ProjectionLayer(params.dim, params.embedding_dim, add_pre_norm=False, add_post_norm=True)
        else:
            self.pre_projector = nn.Identity()
            self.post_projector = nn.Identity()

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(HSTUBlock(layer_id, params))

        self.dropout = nn.Dropout(p=params.dropout)
        self.norm = nn.LayerNorm(params.dim, eps=params.norm_eps)

        self.item_ar_head = SampledARHead(params, self.item_embeddings)
        self.cate_ar_head = FullARHead(params, self.cate_embeddings)
        self.rank_head = RankHead(params)

        # init weights
        self.seed = seed
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(self.seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                historical_item_ids: torch.Tensor,
                historical_cate_ids: torch.Tensor, 
                historical_len: torch.Tensor,
                target_item_id: torch.Tensor,
                target_cate_id: torch.Tensor,
                click_label: torch.Tensor,
                item_ar_labels: torch.Tensor=None,
                cate_ar_labels: torch.Tensor=None,
                **unused):

        batch_size = historical_item_ids.shape[0]
        # expand sequence
        padding = torch.zeros_like(target_item_id).unsqueeze(1)
        input_item_ids = torch.cat([historical_item_ids, padding], axis=1)
        input_cate_ids = torch.cat([historical_cate_ids, padding], axis=1)
        item_ar_labels = torch.cat([item_ar_labels, padding-100], axis=1)
        cate_ar_labels = torch.cat([cate_ar_labels, padding-100], axis=1)
        input_item_ids[torch.arange(batch_size), historical_len] = target_item_id
        input_cate_ids[torch.arange(batch_size), historical_len] = target_cate_id
        # segment id
        segment_ids = torch.zeros_like(input_item_ids)
        segment_ids[torch.arange(batch_size), historical_len] = 1

        seqlen = input_item_ids.shape[1]
        h = self.item_embeddings(input_item_ids) + self.cate_embeddings(input_cate_ids) + self.segment_embeddings(segment_ids)
        h = self.pre_projector(h)
        h = self.dropout(h)

        mask = None
        if self.params.use_causal_mask:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=input_item_ids.device
            )
            mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        h, h_ori = self.post_projector(h), h
        # loss
        if self.params.item_ar_loss_weight > 0:
            _, item_ar_loss = self.item_ar_head(h, item_ar_labels)
            _, cate_ar_loss = self.cate_ar_head(h, cate_ar_labels)
        else:
            item_ar_loss = torch.tensor(0.0)
            cate_ar_loss = torch.tensor(0.0)
        if self.params.rank_loss_weight > 0:
            rank_outputs, rank_loss = self.rank_head(h_ori[torch.arange(batch_size), historical_len], click_label)
        else:
            rank_outputs, rank_loss = torch.zeros_like(click_label, dtype=torch.float), torch.tensor(0.0)
        loss = self.params.item_ar_loss_weight * item_ar_loss + self.params.cate_ar_loss_weight * cate_ar_loss + self.params.rank_loss_weight * rank_loss
        return {
                "loss": loss,
                "item_ar_loss": item_ar_loss,
                "cate_ar_loss": cate_ar_loss,
                "rank_loss": rank_loss,
                "rank_outputs": rank_outputs
            }

from typing import Optional, Tuple
import torch
import math
import gin
from torch import nn
import torch.nn.functional as F
from .args import ModelArgs
from loss import MemoryEfficientV1SampledSoftmaxLoss as SampledSoftmaxLoss, SoftmaxLoss


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     scores = scores + mask  # (bs, n_local_heads, slen, slen)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # output = torch.matmul(scores, xv)  # (bs, n_local_heads, slen, head_dim)
        output = F.scaled_dot_product_attention(xq, xk, xv, None if self.args.use_causal_mask else mask, dropout_p=self.args.dropout, is_causal=self.args.use_causal_mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.dropout(self.attention.forward(
            self.attention_norm(x), freqs_cis, mask
        ))
        out = h + self.dropout(self.feed_forward.forward(self.ffn_norm(h)))
        return out


class SampledARHead(nn.Module):
    def __init__(self, params: ModelArgs, embeddings: torch.nn.Embedding):
        super().__init__()
        self.loss_fn = SampledSoftmaxLoss(projection=embeddings.weight, n_samples=params.n_samples, temperature=params.temperature, l2_norm=params.l2_norm)

    def forward(self, inputs: torch.Tensor, target_ids: torch.Tensor = None, weights: torch.Tensor = None):
        # mask out unrelated samples
        mask = (target_ids != -100).sum(dim=-1).to(torch.bool)
        target_ids = target_ids[mask]
        inputs = inputs[mask]
        if weights is not None:
            weights = weights[mask]
        return (torch.tensor(0), self.loss_fn(inputs, target_ids, weights))


class FullARHead(nn.Module):
    def __init__(self, params: ModelArgs, embeddings: torch.nn.Embedding):
        super().__init__()
        self.loss_fn = SoftmaxLoss(projection=embeddings.weight, temperature=params.temperature, l2_norm=params.l2_norm)

    def forward(self, inputs: torch.Tensor, target_ids: torch.Tensor = None, weights: torch.Tensor = None):
        mask = (target_ids != -100).sum(dim=-1).to(torch.bool)
        target_ids = target_ids[mask]
        inputs = inputs[mask]
        if weights is not None:
            weights = weights[mask]
        return (torch.tensor(0), self.loss_fn(inputs, target_ids, weights))


class RankHead(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.ignore_index = 2 # for ar samples
        self.w1 = nn.Linear(params.dim, params.dim*4, bias=True)
        self.act1 = nn.ReLU()
        self.w2 = nn.Linear(params.dim*4, 1, bias=True)
        self.act2 = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        logits = self.w2(self.act1(self.w1(inputs)))
        logits = logits.squeeze(-1)
        outputs = self.act2(logits)
        if labels is None:
            return (outputs, 0)
        mask = (labels == 0) | (labels == 1)
        loss = self.loss_fn(logits[mask], labels[mask].float())
        return (outputs, loss)


class RankHeadV1(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.ignore_index = 2 # for ar samples
        self.w1 = nn.Linear(params.dim, 1, bias=True)
        self.act1 = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        logits = self.w1(inputs)
        logits = logits.squeeze(-1)
        outputs = self.act1(logits)
        if labels is None:
            return (outputs, 0)
        mask = (labels == 0) | (labels == 1)
        loss = self.loss_fn(logits[mask], labels[mask].float())
        return (outputs, loss)


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, add_pre_norm=True, add_post_norm=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.pre_norm = nn.Identity()
        self.post_norm = nn.Identity()
        if add_pre_norm:
            self.pre_norm = RMSNorm(input_dim)
        if add_post_norm:
            self.post_norm = RMSNorm(output_dim)

    def forward(self, input):
        return self.post_norm(self.linear(self.pre_norm(input)))


@gin.configurable
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, seed: int=0, rank_head_version="v0"):
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
            self.layers.append(TransformerBlock(layer_id, params))

        self.dropout = nn.Dropout(p=params.dropout)
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.item_ar_head = SampledARHead(params, self.item_embeddings)
        self.cate_ar_head = FullARHead(params, self.cate_embeddings)
        if rank_head_version == "v0":
            self.rank_head = RankHead(params)
        elif rank_head_version == "v1":
            self.rank_head = RankHeadV1(params)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
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
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        if self.params.use_causal_mask:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=input_item_ids.device
            )
            mask = torch.triu(mask, diagonal=1).type_as(h)
        else:
            mask = torch.full(
                (batch_size, 1, 1, seqlen), float("-inf"), device=input_item_ids.device
            )
            t = torch.arange(0, seqlen, 1, device=input_item_ids.device)
            l = (historical_len+1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            mask[t < l] = 0.0

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
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

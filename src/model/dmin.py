import torch
import gin
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from .args import ModelArgs


class Dice(nn.Module):
    """
    refer to https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/activations.py
    """
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) * X
        return output


class BehaviorRefinerLayer(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=4, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(BehaviorRefinerLayer, self).__init__()
        self.attention = MultiheadAttention(model_dim,
                                            num_heads=num_heads,
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout = nn.Dropout(net_dropout)
        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm else None

    def forward(self, x, attn_mask=None):
        attn_mask = 1 - attn_mask.float()  # 1 for masked positions in nn.MultiheadAttention
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        s = self.dropout(attn)
        if self.use_residual:
            s += x
        if self.layer_norm is not None:
            s = self.layer_norm(s)
        out = self.ffn(s)
        if self.use_residual:
            out += s
        return out


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """

    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9)  # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


class MultiInterestExtractorLayer(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=4, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True, use_pos_emb=True, pos_emb_dim=8, max_seq_len=10):
        super(MultiInterestExtractorLayer, self).__init__()
        assert model_dim % num_heads == 0, \
            "model_dim={} is not divisible by num_heads={}".format(model_dim, num_heads)
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5
        self.W_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention = ScaledDotProductAttention(attn_dropout)
        self.W_o = nn.ModuleList([nn.Linear(self.head_dim, model_dim, bias=False) for _ in range(num_heads)])
        self.dropout = nn.ModuleList([nn.Dropout(net_dropout) for _ in range(num_heads)]) \
            if net_dropout > 0 else None
        self.layer_norm = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_heads)]) \
            if layer_norm else None
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                                nn.ReLU(),
                                                nn.Linear(ffn_dim, model_dim)) \
                                  for _ in range(num_heads)])
        self.target_attention = nn.ModuleList([TargetAttention(model_dim,
                                                               attention_dropout=attn_dropout,
                                                               use_pos_emb=use_pos_emb,
                                                               pos_emb_dim=pos_emb_dim,
                                                               max_seq_len=max_seq_len) \
                                               for _ in range(num_heads)])

    def forward(self, sequence_emb, target_emb, attn_mask=None, pad_mask=None):
        # linear projection
        query, key, value = torch.chunk(self.W_qkv(sequence_emb), chunks=3, dim=-1)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        attn, _ = self.attention(query, key, value, scale=self.scale, mask=attn_mask)
        # split heads
        attn_heads = torch.chunk(attn, chunks=self.num_heads, dim=1)
        interests = []
        for idx, h_head in enumerate(attn_heads):
            s = self.W_o[idx](h_head.squeeze(1))
            if self.dropout is not None:
                s = self.dropout[idx](s)
            if self.use_residual:
                s += sequence_emb
            if self.layer_norm is not None:
                s = self.layer_norm[idx](s)
            head_out = self.ffn[idx](s)
            if self.use_residual:
                head_out += s
            interest_emb = self.target_attention[idx](head_out, target_emb, mask=pad_mask)
            interests.append(interest_emb)
        return interests


class MlpUint(nn.Module):
    def __init__(self, model_dim, fc_dims, dropout):
        super(MlpUint, self).__init__()
        fc_layers = []
        input_dim = model_dim
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice(fc_dim))
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, attn_input):
        """
        :param attn_input: [B, T, dim]
        :return: [B, T, 1]
        """
        out = self.fc(attn_input)  # [B, T, 1]
        return out


class TargetAttention(nn.Module):
    def __init__(self,
                 model_dim=64,
                 attention_dropout=0.0,
                 use_pos_emb=True,
                 pos_emb_dim=8,
                 max_seq_len=10):
        super(TargetAttention, self).__init__()
        self.model_dim = model_dim
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, pos_emb_dim))
            self.W_proj = nn.Linear(model_dim + pos_emb_dim, model_dim)
        mlp_input_dim = model_dim * 4
        self.attn_mlp = MlpUint(mlp_input_dim, fc_dims=[mlp_input_dim // 2, mlp_input_dim // 4], dropout=attention_dropout)

    def forward(self, sequence_emb, target_emb, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        seq_len = sequence_emb.size(1)
        target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        din_concat = torch.cat([target_emb, sequence_emb, target_emb - sequence_emb,
                                target_emb * sequence_emb], dim=-1)
        attn_score = self.attn_mlp(din_concat.view(-1, 4 * target_emb.size(-1)))
        attn_score = attn_score.view(-1, seq_len)  # b x len
        if mask is not None:
            attn_score = attn_score.masked_fill_(mask.float() == 0, -1.e9)  # fill -inf if mask=0
            attn_score = attn_score.softmax(dim=-1)
        output = (attn_score.unsqueeze(-1) * sequence_emb).sum(dim=1)
        return output


@gin.configurable
class DMIN(nn.Module):
    def __init__(self, params: ModelArgs, seed: int = 0):
        super(DMIN, self).__init__()
        self.item_embeddings = nn.Embedding(params.item_vocab_size, params.dim)
        self.cate_embeddings = nn.Embedding(params.cate_vocab_size, params.dim)
        input_dim = params.dim * 2
        mlp_dims = [params.dim * 2, params.dim]
        self.num_heads = params.n_heads
        
        # behavior sequence modules
        self.behavior_refiner = BehaviorRefinerLayer(input_dim,
                                                     ffn_dim=input_dim * 2,
                                                     num_heads=params.n_heads,
                                                     attn_dropout=params.dropout,
                                                     net_dropout=params.dropout,
                                                     layer_norm=False)
        self.multi_interest_extractor = MultiInterestExtractorLayer(input_dim,
                                                                    ffn_dim=input_dim * 2,
                                                                    num_heads=params.n_heads,
                                                                    attn_dropout=params.dropout,
                                                                    net_dropout=params.dropout,
                                                                    layer_norm=False,
                                                                    use_pos_emb=False,
                                                                    pos_emb_dim=params.pos_emb_dim,
                                                                    max_seq_len=params.max_seq_len)
        fc_layers = []
        fc_input_dim = params.dim * 14
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=params.dropout))
            fc_input_dim = fc_dim
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.mlp = nn.Sequential(*fc_layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
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

    def get_mask(self, x):
        """
        Parameters:
            x: [B, T]
        Returns:
            padding_mask: 0 for masked positions
            attn_mask: 0 for masked positions
        """
        padding_mask = (x == 0)  # 1 for masked positions
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len, 1)
        diag_zeros = ~torch.eye(seq_len, device=x.device).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1)
                .bool().unsqueeze(0).expand_as(attn_mask)
        )
        attn_mask = attn_mask | causal_mask
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(end_dim=1)
        padding_mask, attn_mask = ~padding_mask, ~attn_mask
        return padding_mask, attn_mask

    def forward(self,
                historical_item_ids: torch.Tensor,
                historical_cate_ids: torch.Tensor,
                target_item_id: torch.Tensor,
                target_cate_id: torch.Tensor,
                click_label: torch.Tensor,
                **unused):

        target_item = self.item_embeddings(target_item_id)  # [B, dim]
        target_cate = self.cate_embeddings(target_cate_id)
        user_item_behavior = self.item_embeddings(historical_item_ids)  # [B, T, dim]
        user_cate_behavior = self.cate_embeddings(historical_cate_ids)
        user_seq_rep = torch.cat([user_item_behavior, user_cate_behavior], axis=-1)  # [B, T, 2*dim]
        target_item_rep = torch.cat([target_item, target_cate], axis=-1)  # [B, 2*dim]

        # atten_mask
        pad_mask, attn_mask = self.get_mask(historical_item_ids)
        refined_sequence = self.behavior_refiner(user_seq_rep, attn_mask=attn_mask)
        interests = self.multi_interest_extractor(refined_sequence, target_item_rep,
                                                  attn_mask=attn_mask, pad_mask=pad_mask)

        # sum pooling for user sequences
        seq_mask = (historical_item_ids > 0).unsqueeze(-1).to(target_item.dtype)
        sum_pool_emb = (user_seq_rep * seq_mask).sum(dim=1)

        # concat features
        concat_input = interests
        concat_input = torch.cat(concat_input + [sum_pool_emb, target_item_rep * sum_pool_emb, target_item_rep], dim=1)

        logits = self.mlp(concat_input).squeeze(1)
        loss = self.loss_fn(logits, click_label.float())
        predicts = torch.sigmoid(logits)
        return {
            "loss": loss,
            "rank_loss": loss,
            "rank_outputs": predicts,
            "logits": logits,
            "item_ar_loss": torch.tensor(0.0),
            "cate_ar_loss": torch.tensor(0.0),
        }

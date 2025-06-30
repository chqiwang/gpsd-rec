import torch
import gin
from torch import nn
import torch.nn.functional as F

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

class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim, dropout, fc_dims):
        super(ActivationUnit, self).__init__()
        fc_layers = []
        input_dim = embedding_dim * 4 * 2  # revise     
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice(fc_dim))
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior, mask):
        seq_len = user_behavior.shape[1]
        model_dim = user_behavior.shape[-1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior, 
                                queries * user_behavior], dim=-1)
        attn_input = attn_input.reshape(-1, 4*model_dim)  # [B*L, dim]
        out = self.fc(attn_input)
        out = out.reshape(-1, seq_len, 1)  # [B, L, 1]
        out -= (1-mask) * 1e9
        out = F.softmax(out, dim=1)
        return out

class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim,  dropout):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim = embedding_dim, 
                                          dropout = dropout, fc_dims=[embedding_dim//2, embedding_dim//4])

    def forward(self, query_ad, user_behavior, mask):
        """
        :param query_ad: [B, dim]
        :param user_behavior: [B, L, dim]
        :param mask:
        :return:
        """
        attns = self.active_unit(query_ad, user_behavior, mask)  # [B, L, 1]
        output = user_behavior.mul(attns)  # element-wise
        output = output.sum(dim=1)
        return output


@gin.configurable
class DIN(nn.Module):
    def __init__(self, params: ModelArgs, seed: int=0):
        super(DIN, self).__init__()
        self.item_embeddings = nn.Embedding(params.item_vocab_size, params.dim)
        self.cate_embeddings = nn.Embedding(params.cate_vocab_size, params.dim)
        input_dim = params.dim * 4
        mlp_dims = [params.dim*2, params.dim]
        self.item_attention = AttentionPoolingLayer(params.dim, params.dropout)
        self.cate_attention = AttentionPoolingLayer(params.dim, params.dropout)
        fc_layers = []
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=params.dropout))
            input_dim = fc_dim
        fc_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*fc_layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        # init weights
        self.seed = seed
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(self.seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.8)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
            historical_item_ids: torch.Tensor,
            historical_cate_ids: torch.Tensor,
            target_item_id: torch.Tensor,
            target_cate_id: torch.Tensor,
            click_label: torch.Tensor,
            **unused):
        target_item = self.item_embeddings(target_item_id).unsqueeze(1)
        target_cate = self.cate_embeddings(target_cate_id).unsqueeze(1)
        mask = (historical_item_ids > 0).unsqueeze(-1).to(target_item.dtype)
        user_item_behavior = self.item_embeddings(historical_item_ids)
        user_item_behavior = user_item_behavior.mul(mask)
        user_cate_behavior = self.cate_embeddings(historical_cate_ids)
        user_cate_behavior = user_cate_behavior.mul(mask)
        
        # first concat
        user_behavior = torch.concat([user_item_behavior, user_cate_behavior], dim=-1)
        target_info = torch.concat([target_item, target_cate], dim=-1)
        # attention layer
        user_interest = self.item_attention(target_info, user_behavior, mask)
        concat_input = torch.cat([user_interest, target_item.squeeze(1), target_cate.squeeze(1)], dim = 1)
        logits = self.mlp(concat_input).squeeze(1)
        loss = self.loss_fn(logits, click_label.float())
        predicts = torch.sigmoid(logits)
        return {
            "loss": loss,
            "rank_loss": loss,
            "rank_outputs": predicts,
            "item_ar_loss": torch.tensor(0.0),
            "cate_ar_loss": torch.tensor(0.0),
        }
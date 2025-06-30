import gin
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .args import ModelArgs

# some of the code is taken from https://github.com/clabrugere/wukong-recommendation

class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int | None = None,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_hidden))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, dim_hidden))

        super().__init__(*layers)


class LinearCompressBlock(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)
        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight
        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)
        return outputs


class FactorizationMachineBlock(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((num_emb_in, rank)))
        self.norm = nn.LayerNorm(num_emb_in * rank)
        self.mlp = MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)
        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.weight
        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = torch.bmm(inputs, outputs)
        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = outputs.view(-1, self.num_emb_in * self.rank)
        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))
        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.view(-1, self.num_emb_out, self.dim_emb)

        return outputs


class ResidualProjection(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)
        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight
        # # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)
        return outputs


class WukongLayer(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)
        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = torch.concat((fmb, lcb), dim=1)
        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))
        return outputs

@gin.configurable
class Wukong(nn.Module):
    def __init__(
        self,
        params: ModelArgs,
        num_emb_lcb:int=16,
        num_emb_fmb:int=16,
        num_layers=3,
        rank_fmb:int=24,
        num_hidden_wukong=2,
        dim_hidden_wukong=512,
        num_hidden_head=2,
        dim_hidden_head=512,
        dim_emb=128,
        dim_output=1,
        dropout=0.0,
        seed=0,
    ) -> None:
        super().__init__()
        num_emb_in = params.max_seq_len * 2 + 2
        self.max_seq_len = params.max_seq_len
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.dim_emb = dim_emb
        self.seed = seed

        self.item_embeddings = torch.nn.Embedding(params.item_vocab_size, dim_emb)
        self.cate_embeddings = torch.nn.Embedding(params.cate_vocab_size, dim_emb)

        self.interaction_layers = nn.Sequential()
        for _ in range(num_layers):
            self.interaction_layers.append(
                WukongLayer(
                    num_emb_in,
                    dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_wukong,
                    dim_hidden_wukong,
                    dropout,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb

        self.projection_head = MLP(
            (num_emb_lcb + num_emb_fmb) * dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
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
                target_item_id: torch.Tensor,
                target_cate_id: torch.Tensor,
                click_label: torch.Tensor, **unused):
        if historical_item_ids.size(1) > self.max_seq_len:
             historical_item_ids = historical_item_ids[:, :self.max_seq_len]
             historical_cate_ids = historical_cate_ids[:, :self.max_seq_len]
        else:
            historical_item_ids = F.pad(historical_item_ids, [0,self.max_seq_len-historical_item_ids.size(1),0,0])
            historical_cate_ids = F.pad(historical_cate_ids, [0,self.max_seq_len-historical_cate_ids.size(1),0,0])

        his_item_embeddings = self.item_embeddings(historical_item_ids.flip(1))
        his_cate_embeddings = self.cate_embeddings(historical_cate_ids.flip(1))
        target_item_embedding = self.item_embeddings(target_item_id)
        target_cate_embedding = self.item_embeddings(target_cate_id)
        outputs = torch.concat([his_item_embeddings, his_cate_embeddings, target_item_embedding.unsqueeze(dim=1), target_cate_embedding.unsqueeze(dim=1)], dim=1)
        outputs = self.interaction_layers(outputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb)
        outputs = self.projection_head(outputs)

        logits = outputs.squeeze(1)
        loss = self.loss_fn(logits, click_label.float())
        predicts = torch.sigmoid(logits)
        return {
                "loss": loss,
                "rank_loss": loss,
                "rank_outputs": predicts,
                "item_ar_loss": torch.tensor(0.0),
                "cate_ar_loss": torch.tensor(0.0),
            }

import torch
import gin
from torch import nn
import torch.nn.functional as F

from .args import ModelArgs

def att_mlp_blocks(input_dim, mlp_dims, dropout, output_dim=1):
    fc_layers = []
    for fc_dim in mlp_dims:
        fc_layers.append(nn.Linear(input_dim, fc_dim))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(p=dropout))
        input_dim = fc_dim
    if output_dim == 1:
        fc_layers.append(nn.Linear(input_dim, 1))
    else:
        fc_layers.append(nn.Linear(input_dim, output_dim))
    mlp = nn.Sequential(*fc_layers)
    return mlp


class LogisticRegression(nn.Module):
    def __init__(self, feature_num, emb_dim):
        super(LogisticRegression, self).__init__()
        input_dim = feature_num * emb_dim
        self.mlp = att_mlp_blocks(feature_num * emb_dim, mlp_dims=[input_dim//2, input_dim//4], dropout=0.0, output_dim=emb_dim)

    def forward(self, X):
        """
        :param X: b x N x dim
        :return:
        """
        embed_weights = torch.reshape(X, (-1, X.size(-2) * X.size(-1)))  # [b, Nxdim]
        output = self.mlp(embed_weights)
        return output


class InnerProductInteraction(nn.Module):
    """ output: product_sum (bs x 1),
                bi_interaction (bs * dim),
                inner_product (bs x f^2/2),
                elementwise_product (bs x f^2/2 x emb_dim)
    """
    def __init__(self, num_fields, output="product_sum"):
        super(InnerProductInteraction, self).__init__()
        self._output_type = output
        if output not in ["product_sum", "bi_interaction", "inner_product", "elementwise_product"]:
            raise ValueError("InnerProductInteraction output={} is not supported.".format(output))
        if output == "inner_product":
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.triu_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).bool(),
                                          requires_grad=False)
        elif output == "elementwise_product":
            self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)

    def forward(self, feature_emb):
        """
        :param feature_emb: b x N x dim
        :return:
        """
        if self._output_type in ["product_sum", "bi_interaction"]:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb ** 2, dim=1) # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "bi_interaction":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            triu_values = torch.masked_select(inner_product_matrix, self.triu_mask)
            return triu_values.view(-1, self.interaction_units)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.triu_index[0])
            emb2 = torch.index_select(feature_emb, 1, self.triu_index[1])
            return emb1 * emb2


class FactorizationMachine(nn.Module):
    def __init__(self, feature_num_fields, feature_dim):
        """
        :param feature_num_fields:  feature number
        :param feature_dim: feature dim
        """
        super(FactorizationMachine, self).__init__()
        self.fm_layer = InnerProductInteraction(feature_num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_num_fields, feature_dim)


    def forward(self, feature_emb):
        """
        :param feature_emb: [b, N, dim]
        :return:
        """
        lr_out = self.lr_layer(feature_emb)
        fm_out = self.fm_layer(feature_emb)
        output = fm_out + lr_out
        return output


@gin.configurable
class DeepFM(nn.Module):
    def __init__(self, params: ModelArgs, seed: int=0):
        super(DeepFM, self).__init__()
        self.item_embeddings = nn.Embedding(params.item_vocab_size, params.dim)
        self.cate_embeddings = nn.Embedding(params.cate_vocab_size, params.dim)
        feature_num, feature_dim = 4, params.dim  # target: cate, item; sequence: cate, item;
        self.fm_layer = FactorizationMachine(feature_num, feature_dim)
        self.output_proj = nn.Linear(feature_dim, 1)
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

    def forward(self,
            historical_item_ids: torch.Tensor,
            historical_cate_ids: torch.Tensor,
            target_item_id: torch.Tensor,
            target_cate_id: torch.Tensor,
            click_label: torch.Tensor,
            **unused):
        target_item = self.item_embeddings(target_item_id)
        target_cate = self.cate_embeddings(target_cate_id)
        mask = (historical_item_ids > 0).unsqueeze(-1).to(target_item.dtype)
        user_item_behavior = self.item_embeddings(historical_item_ids)
        user_item_behavior = user_item_behavior.mul(mask)  # [b, L, dim]
        user_cate_behavior = self.cate_embeddings(historical_cate_ids)
        user_cate_behavior = user_cate_behavior.mul(mask)

        # stack features
        concat_features = torch.stack([target_item, target_cate, torch.mean(user_item_behavior, dim=1), torch.mean(user_cate_behavior, dim=1)], dim=1)  # [b, N, dim]
        fm_output = self.fm_layer(concat_features)
        logits = torch.squeeze(self.output_proj(fm_output), dim=-1)
        loss = self.loss_fn(logits, click_label.float())
        predicts = torch.sigmoid(logits)

        return {
            "loss": loss,
            "rank_loss": loss,
            "rank_outputs": predicts,
            "item_ar_loss": torch.tensor(0.0),
            "cate_ar_loss": torch.tensor(0.0),
        }
import torch
import gin
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
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
        input_dim = embedding_dim * 4
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice(fc_dim))
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior, mask):
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior,
                                queries * user_behavior], dim=-1)
        out = self.fc(attn_input)
        out -= (1 - mask) * 1e4
        out = F.softmax(out, dim=1)
        return out


class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim=embedding_dim,
                                          dropout=dropout, fc_dims=[embedding_dim // 2, embedding_dim // 4])

    def forward(self, query_ad, user_behavior, mask):
        attns = self.active_unit(query_ad, user_behavior, mask)
        output = user_behavior.mul(attns)
        output = output.sum(dim=1)
        return output


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


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, attention_type="bilinear_attention", use_attention_softmax=True, dropout=0.0):
        super(AttentionLayer, self).__init__()
        self.attention_type = attention_type
        self.use_attention_softmax = use_attention_softmax
        self.W_kernel = nn.Parameter(torch.eye(model_dim))
        self.attn_mlp = MlpUint(model_dim, fc_dims=[model_dim // 2, model_dim // 4], dropout=dropout)
        self.model_dim = model_dim  # model_dim = dim * 8

    def forward(self, sequence_emb, target_emb, mask=None):
        """
        :param sequence_emb: [B, T, dim]
        :param target_emb: [B, dim]
        :param mask: [B, T]
        :return:  [B, T]
        """
        seq_len = sequence_emb.size(1)
        attn_score = sequence_emb @ target_emb.unsqueeze(-1)
        if self.attention_type == "bilinear_attention":
            attn_score = (sequence_emb @ self.W_kernel) @ target_emb.unsqueeze(-1)
        elif self.attention_type == "din_attention":
            target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, dim]
            din_concat = torch.cat([target_emb, sequence_emb, target_emb - sequence_emb,
                                    target_emb * sequence_emb], dim=-1)  # [B, T, dim]
            din_tensor = din_concat.reshape(-1, self.model_dim)  # [B, T, 64 * 2 * 4]
            attn_score = self.attn_mlp(din_tensor)  # [B*T, 1]

        attn_score = attn_score.reshape(-1, seq_len)  # [B, T]
        if mask is not None:
            attn_score = attn_score * mask.float()
        if self.use_attention_softmax:
            if mask is not None:
                attn_score += -1.e9 * (1 - mask.float())
            attn_score = attn_score.softmax(dim=-1)
        return attn_score


class AGRUCell(nn.Module):
    r"""AGRUCell with attentional update gate
        Reference: GRUCell from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hx, attn):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hx)

        i_u, i_r, i_n = gate_x.chunk(3, 1)
        h_u, h_r, h_n = gate_h.chunk(3, 1)

        reset_gate = F.sigmoid(i_r + h_r)
        new_gate = F.tanh(i_n + reset_gate * h_n)
        hy = hx + attn.view(-1, 1) * (new_gate - hx)
        return hy


class AUGRUCell(nn.Module):
    r"""AUGRUCell with attentional update gate
        Reference: GRUCell from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hx, attn):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hx)

        i_u, i_r, i_n = gate_x.chunk(3, 1)
        h_u, h_r, h_n = gate_h.chunk(3, 1)

        update_gate = torch.sigmoid(i_u + h_u)
        update_gate = update_gate * attn.unsqueeze(-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = hx + update_gate * (new_gate - hx)
        return hy


class DynamicGRU(nn.Module):
    r"""DynamicGRU with GRU, AIGRU, AGRU, and AUGRU choices
        Reference: https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/prediction_flow/pytorch/nn/rnn.py
    """

    def __init__(self, input_size, hidden_size, bias=True, gru_type='AUGRU'):
        super(DynamicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_type = gru_type
        if gru_type == "AUGRU":
            self.gru_cell = AUGRUCell(input_size, hidden_size, bias=bias)
        elif gru_type == "AGRU":
            self.gru_cell = AGRUCell(input_size, hidden_size, bias=bias)

    def forward(self, packed_seq_emb, attn_score=None, h=None):
        assert isinstance(packed_seq_emb, PackedSequence) and isinstance(attn_score, PackedSequence), \
            "DynamicGRU supports only `PackedSequence` input."
        x, batch_sizes, sorted_indices, unsorted_indices = packed_seq_emb
        attn, _, _, _ = attn_score

        if h == None:
            h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
        output_h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
        outputs = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        start = 0
        for batch_size in batch_sizes:
            _x = x[start: start + batch_size]
            _h = h[:batch_size]
            _attn = attn[start: start + batch_size]
            h = self.gru_cell(_x, _h, _attn)
            outputs[start: start + batch_size] = h
            output_h[:batch_size] = h
            start += batch_size

        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices), \
               output_h[unsorted_indices]


@gin.configurable
class DIEN(nn.Module):
    """
    reference to https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DIEN/src/DIEN.py
    """

    def __init__(self, params: ModelArgs, seed: int = 0, gru_type: str = 'AGRU'):
        super(DIEN, self).__init__()
        self.item_embeddings = nn.Embedding(params.item_vocab_size, params.dim)
        self.cate_embeddings = nn.Embedding(params.cate_vocab_size, params.dim)
        input_dim = params.dim * 8
        mlp_dims = [params.dim * 2, params.dim]
        self.item_attention = AttentionPoolingLayer(params.dim, params.dropout)
        self.cate_attention = AttentionPoolingLayer(params.dim, params.dropout)
        self.gru_type = gru_type

        # interest extraction
        model_dim = params.dim * 2  # item_id and cate_id
        self.extraction_module = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        # interest evolving
        if gru_type in ["AGRU", "AUGRU"]:
            self.evolving_module = DynamicGRU(model_dim, model_dim, gru_type=gru_type)
        else:
            self.evolving_module = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)

        # attention layer
        if gru_type in ["AIGRU", "AGRU", "AUGRU"]:
            self.attention_module = AttentionLayer(params.dim * 8, dropout=params.dropout, attention_type=params.attention_type)

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
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def interest_extraction(self, sequence_emb, mask):
        """
        :param sequence_emb: [B, T, dim]
        :param mask: [B, T]
        :return: PackedSequence, [B, T, dim]
        """
        seq_lens = mask.sum(dim=1).cpu()  # [B]
        packed_seq = pack_padded_sequence(sequence_emb,
                                          seq_lens,
                                          batch_first=True,
                                          enforce_sorted=False)
        packed_interests, _ = self.extraction_module(packed_seq)
        interest_emb, _ = pad_packed_sequence(packed_interests,
                                              batch_first=True,
                                              padding_value=0.0,
                                              total_length=mask.size(1))
        return packed_interests, interest_emb

    def interest_evolution(self, packed_interests, interest_emb, target_emb, mask):
        """
        :param packed_interests: PackedSequence
        :param interest_emb: [B, T, 2*dim]
        :param target_emb: [B, dim]
        :param mask: [B, T]
        :return: [B, dim]
        """
        if self.gru_type == "GRU":
            _, h_out = self.evolving_module(packed_interests)  # [B, T, dim]
        else:
            attn_scores = self.attention_module(interest_emb, target_emb, mask)  # [B, T]
            seq_lens = mask.sum(dim=1).cpu()  # [B]
            if self.gru_type == "AIGRU":
                packed_inputs = pack_padded_sequence(interest_emb * attn_scores,
                                                     seq_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, h_out = self.evolving_module(packed_inputs)
            else:
                packed_scores = pack_padded_sequence(attn_scores,
                                                     seq_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, h_out = self.evolving_module(packed_interests, packed_scores)

        return h_out.squeeze()

    def get_unmasked_tensor(self, h, non_zero_mask):
        out = torch.zeros([non_zero_mask.size(0)] + list(h.shape[1:]), device=h.device)
        out[non_zero_mask] = h
        return out

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

        # DIEN
        pad_mask = (historical_item_ids > 0).to(target_item.dtype)  # [B, T]
        non_zero_mask = pad_mask.sum(dim=1) > 0  # [B]
        packed_interests, interest_emb = self.interest_extraction(user_seq_rep[non_zero_mask],
                                                                  pad_mask[non_zero_mask])  # rnn layer

        h_out = self.interest_evolution(packed_interests, interest_emb, target_item_rep[non_zero_mask],
                                        pad_mask[non_zero_mask])  # merge layer
        final_out = self.get_unmasked_tensor(h_out, non_zero_mask)

        concat_emb = [final_out]

        seq_mask = (historical_item_ids > 0).unsqueeze(-1).to(target_item.dtype)
        sum_pool_emb = (user_seq_rep * seq_mask).sum(dim=1)
        concat_emb += [sum_pool_emb, target_item_rep * sum_pool_emb, target_item_rep]

        # concat features
        concat_input = torch.cat(concat_emb, dim=1)

        # DNN layer
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

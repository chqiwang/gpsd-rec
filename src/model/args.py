from dataclasses import dataclass
import gin

@gin.configurable
@dataclass
class ModelArgs:
    dim: int = 64
    embedding_dim: int = None # if None, dim should be take as embedding dim
    n_layers: int = 4
    n_heads: int = 4
    item_vocab_size: int = None
    cate_vocab_size: int = None
    segment_vocab_size: int = 2
    multiple_of: int = 32  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 200
    dropout:int = 0.0
    rank_loss_weight:float = 1.0
    use_causal_mask:bool=True
    # AR
    n_samples:int = 4096
    temperature:float = 0.05
    l2_norm:bool = True
    item_ar_loss_weight:float = 0.0
    cate_ar_loss_weight:float = 0.0
    # Other
    attention_type:str = "bilinear_attention"  # bilinear_attention, din_attention, None
    pos_emb_dim:int = 32

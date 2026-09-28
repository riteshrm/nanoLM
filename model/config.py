from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int

    mlp_ratio: float = 4.0

    attention_type: str = "mha"
    mlp_type: str = "swiglu"

    use_rope: bool = True
    norm_type: str = "rmsnorm"

    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2
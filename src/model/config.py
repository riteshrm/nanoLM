from dataclasses import dataclass, fields

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
    pe_type: str= 'sincos'

    use_rope: bool = True
    norm_type: str = "rmsnorm"

    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
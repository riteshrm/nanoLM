import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .mlp import MLP
from .normalization import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.attention_type=='mha':
            self.attention = MultiHeadAttention(args)
        self.norm1 = LayerNorm(args.d_model)
        self.mlp = MLP(d_model=args.d_model, mlp_ratio=args.mlp_ratio)
        self.norm2 = LayerNorm(args.d_model)

    def forward(self, x, rotary_emb=None):
        x = x + self.attention(self.norm1(x), rotary_emb)
        x = x + self.mlp(self.norm2(x))
        return x
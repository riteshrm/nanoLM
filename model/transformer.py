import torch
import torch.nn as nn
from .block import TransformerBlock
from .position_embedding import sin_cos_pe

class LanguageModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.lm_head = nn.Linear(args.d_model, args.vocab_size)

    def forward(self, x):
        token_embedding = self.embedding(x)
        poisition_embedding = sin_cos_pe(d_model=self.d_model, pos=x)

        x = token_embedding + poisition_embedding

        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits
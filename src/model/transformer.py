import torch
import torch.nn as nn
from .block import TransformerBlock
from .position_embedding import sin_cos_pe, rope
from .config import ModelConfig
from.normalization import LayerNorm
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
from dataclasses import asdict
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = LayerNorm(self.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        token_embedding = self.embedding(x)
        poisition_embedding=None
        if self.config.pe_type=='sincos':
            poisition_embedding = sin_cos_pe(pos=x, d_model=self.d_model)
            x = token_embedding + poisition_embedding
            poisition_embedding=None
        else:
            poisition_embedding = rope(pos=x, d_model=self.d_model)
            x = token_embedding

        for block in self.blocks:
            x = block(x, poisition_embedding)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens=100):
        prompt_len = idx.shape[1]
        for _ in range(max_new_tokens):
            # crop the index till block_size
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            # Focus on last time step
            logits = logits[:,-1,:] # [B, C]
            # apply softmax to get the probabilty
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to the running sequnce
            if idx_next[0]>=512:
                            break
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]

        return idx[:, prompt_len:]
    
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        return cls(config) 

    def save_pretrained(self, path, state_dict=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        sd = state_dict if state_dict is not None else self.state_dict()
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
        save_file(sd, str(path / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path, device="cpu", strict=True):
        path = Path(path)
        config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = cls(config)
        model.load_state_dict(load_file(str(path / "model.safetensors"), device=device), strict=strict)
        return model


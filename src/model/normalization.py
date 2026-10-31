import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0) # We are considering population variance not sample
        

        x = (x - mean) / torch.sqrt(var + self.eps)

        return x * self.gamma + self.beta
# class RMSNorm(...)
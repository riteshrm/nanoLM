import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=d_model, out_features=int(mlp_ratio*d_model))
        self.mlp2 = nn.Linear(in_features=int(mlp_ratio*d_model), out_features=d_model)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        return x
    
# class SwiGLU(nn.Module)
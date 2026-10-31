import torch

def sin_cos_pe(pos, d_model):
    bsz, n_tokens = pos.shape
    pos = pos.reshape(bsz, n_tokens, -1)
    div_term = torch.tensor([10000**(2*i/d_model) for i in range(d_model//2)]).to(pos.device)
    sinPE = torch.sin(pos/div_term)
    cosPE = torch.cos(pos/div_term)
    combinedPE = torch.zeros(bsz, n_tokens, d_model).to(pos.device)
    combinedPE[:, :, 0::2] = sinPE
    combinedPE[:, :, 1::2] = cosPE

    return combinedPE

def rope(pos, d_model):
    theta = torch.tensor([10000**(-2*(i-1)/d_model) for i in range(1, d_model//2+1)]).to(pos.device)
    bsz, seq_len= pos.shape
    cos = torch.cos(pos[:, :, None]* theta[None, :]).view(bsz, seq_len, d_model//2)
    sin = torch.sin(pos[:, :, None]* theta[None, :]).view(bsz, seq_len, d_model//2)
    row1 = torch.stack([cos, -sin], dim=-1)
    row2 = torch.stack([sin, cos], dim=-1)

    RTheta = torch.stack([row1, row2], dim=-2)
    return RTheta
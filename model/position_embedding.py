import torch

def sin_cos_pe(d_model, pos):
    bsz, n_tokens = pos.shape
    pos = pos.reshape(bsz, n_tokens, -1)
    div_term = torch.tensor([10000**(2*i/d_model) for i in range(d_model//2)]).to(pos.device)
    sinPE = torch.sin(pos/div_term)
    cosPE = torch.cos(pos/div_term)
    combinedPE = torch.zeros(bsz, n_tokens, d_model).to(pos.device)
    combinedPE[:, :, 0::2] = sinPE
    combinedPE[:, :, 1::2] = cosPE

    return combinedPE
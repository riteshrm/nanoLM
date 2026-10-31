import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.q = nn.Linear(in_features=args.d_model, out_features=args.n_heads*args.d_model)
        self.k = nn.Linear(in_features=args.d_model, out_features=args.n_heads*args.d_model)
        self.v = nn.Linear(in_features=args.d_model, out_features=args.n_heads*args.d_model)
        self.register_buffer("trill", torch.tril(torch.ones(args.max_seq_len, args.max_seq_len)))
        self.proj_out = nn.Linear(in_features=args.n_heads*args.d_model, out_features=args.d_model)

    def forward(self, x, rotary_emb=None):
        bsz, seq_len, d_model = x.shape
        query = self.q(x).reshape(bsz, seq_len, -1, d_model).transpose(1, 2)
        key = self.k(x).reshape(bsz, seq_len, -1, d_model).transpose(1, 2)
        value = self.v(x).reshape(bsz, seq_len, -1, d_model).transpose(1, 2)
        if rotary_emb is not None:
            rotary_emb = rotary_emb.unsqueeze(1)
            query = query.reshape(bsz, -1, seq_len, d_model//2, 2, 1)
            key = key.reshape(bsz, -1, seq_len, d_model//2, 2, 1)
            query =  torch.matmul(rotary_emb, query).reshape(bsz, -1, seq_len, d_model)
            key =  torch.matmul(rotary_emb, key).reshape(bsz, -1, seq_len, d_model)
        score = torch.matmul(query, key.transpose(-2, -1))/(d_model**0.5)
        score = score.masked_fill(self.trill[:seq_len, :seq_len]==0, float('-inf'))
        attention_score = F.softmax(score, dim=-1)
        out = torch.matmul(attention_score, value).transpose(1, 2).reshape(bsz, seq_len, -1)
        final_out = self.proj_out(out)
        return final_out


# class GroupedQueryAttention(...)
# class MultiHeadLatentAttention(...)
# class LinearAttention(...)
# class MultiQueryAttention(...)
import math

import torch
from torch import nn


# class SparseMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.1, attn_span=100):
#         super(SparseMultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.attn_span = attn_span
#         self.head_dim = embed_dim // num_heads
#         if self.head_dim * num_heads != self.embed_dim:
#             raise ValueError("embed_dim must be divisible by num_heads")
#         self.query_ff = nn.Linear(embed_dim, embed_dim)
#         self.key_ff = nn.Linear(embed_dim, embed_dim)
#         self.value_ff = nn.Linear(embed_dim, embed_dim)
#         self.out_ff = nn.Linear(embed_dim, embed_dim)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, query, key, value, **kwargs):
#         # pytorch sparse tensors still under active development, so expect changes soon
#         n = query.size(0)
#         query = self.query_ff(query).view(n, -1, self.head_dim).transpose(0, 1)
#         key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
#         key = torch.cat([key, torch.zeros(key.size(0), 1, key.size(2), device=key.device)], dim=1)
#         value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
#         value = torch.cat([value, torch.zeros(value.size(0), 1, value.size(2), device=value.device)], dim=1)
#         rows = torch.arange(n, device=query.device).repeat(2 * self.attn_span + 1, 1).transpose(0, 1).flatten()
#         cols = torch.cat([torch.arange(i - self.attn_span, i + self.attn_span + 1, device=query.device) for i in range(n)])
#         out_of_bounds = (cols < 0) | (cols >= n)
#         cols[out_of_bounds] = n
#         idxs = torch.stack([rows, cols])
#         vals = (query[:, rows, :] * key[:, cols, :]).sum(-1) / math.sqrt(n)
#         vals[:, out_of_bounds] = -float("inf")
#         del query, key, out_of_bounds
#         vals = torch.dropout(torch.softmax(vals.view(-1, n, 2 * self.attn_span + 1), dim=-1), self.dropout, self.training).view(-1, idxs.size(1))
#         attn_matrix = [torch.sparse.FloatTensor(idxs, val, (n, n + 1)) for val in vals]
#         out = self.out_ff(torch.stack([attn @ val for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, -1, self.embed_dim))
#         return out

class SparseKernelMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, distribution, num_samples=1000):
        super(SparseKernelMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.distribution = distribution
        self.num_samples = num_samples
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.query_ff = nn.Linear(embed_dim, embed_dim)
        self.key_ff = nn.Linear(embed_dim, embed_dim)
        self.value_ff = nn.Linear(embed_dim, embed_dim)
        self.out_ff = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, **kwargs):
        # pytorch sparse tensors still under active development, so expect changes soon
        n = query.size(0)
        device = query.device
        query = self.query_ff(query).view(n, -1, self.head_dim).transpose(0, 1)
        key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
        value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
        samples = self.distribution.sample((n, self.num_samples)).round().long()
        samples = (samples + torch.arange(n, device=device).unsqueeze(1)).clamp(0, n - 1)
        cols = [samples[i].unique() for i in range(n)]
        del samples
        rows = [torch.tensor(i, device=device).repeat(cols[i].size(0)) for i in range(n)]
        vals = [torch.softmax((query[:, rows[i], :] * key[:, cols[i], :]).sum(-1) / math.sqrt(n), dim=-1) for i in range(n)]
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals, dim=-1)
        idxs = torch.stack([rows, cols])
        del rows, cols, query, key
        attn_matrix = [torch.sparse.FloatTensor(idxs, val, (n, n)) for val in vals]
        out = self.out_ff(torch.stack([attn @ val for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, -1, self.embed_dim))
        return out

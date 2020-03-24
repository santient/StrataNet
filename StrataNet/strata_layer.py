import math

import torch
from torch import nn
from torch.utils import checkpoint


EPS = 1e-10

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x, idx=0):
        pe = torch.zeros(x.shape[0], x.shape[2], device=x.device)
        position = torch.arange(idx, idx + x.shape[0], device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device) * -(math.log(10000.0) / x.shape[2]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return x + pe

class HierarchyFunction(nn.Module):
    def __init__(self):
        super(HierarchyFunction, self).__init__()

    def forward(self, vertical, horizontal, mean, idx):
        mask = torch.zeros_like(vertical)
        mask[vertical <= 0] = -float("inf")
        return torch.softmax(vertical * torch.exp(-(idx - mean)**2 / (2 * horizontal**2 + EPS)) + mask, dim=0)

class StrataLayer(nn.Module):
    def __init__(self, input_dim, d_model, nhead=8, dim_feedforward=1024, dropout=0.1, num_layers=6, output_hierarchy=False, output_hierarchy_dim=None):
        super(StrataLayer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_hierarchy = output_hierarchy
        self.output_hierarchy_dim = output_hierarchy_dim
        self.input_transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            nn.Linear(input_dim, d_model))
        self.output_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers)
        self.positional_enc = PositionalEncoding()
        self.hierarchy_func = HierarchyFunction()
        if output_hierarchy:
            if output_hierarchy_dim is None:
                raise ValueError("output_hierarchy_dim must be specified if output_hierarchy is true")
            self.vertical_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers)
            self.horizontal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers)
            self.vertical_ff = nn.Linear(d_model, output_hierarchy_dim)
            self.horizontal_ff = nn.Linear(d_model, output_hierarchy_dim)

    def forward(self, inputs, length, block_size, hierarchy_params=None):
        if hierarchy_params is None:
            hierarchy_params = (torch.zeros(inputs.shape[0], inputs.shape[1], self.d_model, device=inputs.device),
                                torch.zeros(inputs.shape[0], inputs.shape[1], self.d_model, device=inputs.device))
        vertical, horizontal = hierarchy_params
        window = []
        pos = 0.0
        in_idx = 0
        out = []
        if self.output_hierarchy:
            out_vertical = []
            out_horizontal = []
        for out_idx in range(length):
            low = out_idx - block_size // 2
            high = out_idx + block_size // 2
            while len(window) > 0 and window[0][0] < out_idx:
                del window[0]
            while in_idx < inputs.shape[0] and high >= pos:
                low_pos = max(0, int(pos) - block_size // 2)
                high_pos = min(length, int(pos) + block_size // 2)
                enc = self.positional_enc(inputs[in_idx].repeat(high_pos - low_pos, 1, 1), low_pos)
                if torch.is_grad_enabled():
                    enc = checkpoint.checkpoint(self.input_transformer, enc)
                else:
                    enc = self.input_transformer(enc)
                v_in = vertical[in_idx]
                h_in = horizontal[in_idx]
                if torch.is_grad_enabled():
                    x_out = checkpoint.checkpoint(self.output_transformer, enc)
                else:
                    x_out = self.output_transformer(enc)
                if self.output_hierarchy:
                    if torch.is_grad_enabled():
                        v_out = checkpoint.checkpoint(self.vertical_transformer, enc)
                        h_out = checkpoint.checkpoint(self.horizontal_transformer, enc)
                    else:
                        v_out = self.vertical_transformer(enc)
                        h_out = self.horizontal_transformer(enc)
                    window.append((high_pos, x_out, v_in, h_in, v_out, h_out))
                else:
                    window.append((high_pos, x_out, v_in, h_in))
                del enc
                in_idx += 1
                if inputs.shape[0] > 1:
                    pos += (length - 1) / (inputs.shape[0] - 1)
            if len(window) > 0:
                v = torch.stack([item[2] for item in window])
                h = (length / inputs.shape[0]) * torch.stack([item[3] for item in window])
                m = torch.tensor([item[0] - block_size // 2 for item in window], device=inputs.device).view(-1, 1, 1)
                hf = self.hierarchy_func(v, h, m, out_idx)
                x = torch.stack([item[1][out_idx - item[0]] for item in window])
                outx = torch.sum(hf * x, dim=0)
                out.append(outx)
                del v, h, m, x
                if self.output_hierarchy:
                    v = torch.stack([item[4][out_idx - item[0]] for item in window])
                    h = torch.stack([item[5][out_idx - item[0]] for item in window])
                    outv = torch.sum(hf * v, dim=0)
                    out_vertical.append(outv)
                    outh = torch.sum(hf * h, dim=0)
                    out_horizontal.append(outh)
                    del v, h
                del hf
            else:
                raise RuntimeError("block_size too low to capture full sequence")
        out = torch.stack(out)
        if self.output_hierarchy:
            out_vertical = self.vertical_ff(torch.stack(out_vertical))
            out_horizontal = self.horizontal_ff(torch.stack(out_horizontal))
            return out, (out_vertical, out_horizontal)
        else:
            return out

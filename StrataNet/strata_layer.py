import math

import torch
from torch import nn


EPS = 1e-10

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x, idx=0):
        pe = torch.zeros(x.shape[0], x.shape[2], device=x.device)
        position = torch.arange(idx, idx + x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.shape[2], 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / x.shape[2]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return x + pe

class HierarchyFunction(nn.Module):
    def __init__(self):
        super(HierarchyFunction, self).__init__()

    def forward(self, vertical, horizontal, mean, idx):
        return torch.softmax(vertical**2 * torch.exp(-(idx - mean)**2 / (horizontal**2 + EPS)), dim=0)

class StrataLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.1, num_layers=6, output_hierarchy=True):
        super(StrataLayer, self).__init__()
        self.output_hierarchy = output_hierarchy
        self.output_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers)
        if output_hierarchy:
            self.vertical_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers)
            self.horizontal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers)
        self.pe = PositionalEncoding()
        self.hf = HierarchyFunction()

    def forward(self, inputs, length, block_size, hierarchy=None):
        if hierarchy is None:
            hierarchy = (torch.zeros_like(inputs), torch.zeros_like(inputs))
        vertical, horizontal = hierarchy
        window = []
        pos = 0.0
        in_idx = 0
        out = torch.zeros(length, inputs.shape[1], inputs.shape[2], device=inputs.device)
        if self.output_hierarchy:
            out_vertical = torch.zeros_like(out)
            out_horizontal = torch.zeros_like(out)
        for out_idx in range(length):
            low = out_idx - block_size // 2
            high = out_idx + block_size // 2
            while len(window) > 0 and window[0][0] < out_idx:
                del window[0]
            while in_idx < inputs.shape[0] and high >= pos:
                low_pos = int(pos) - block_size // 2
                high_pos = int(pos) + block_size // 2
                enc = self.pe(inputs[in_idx].repeat(block_size, 1, 1), low_pos)
                v_in = vertical[in_idx]
                h_in = horizontal[in_idx]
                x_out = self.output_transformer(enc)
                if self.output_hierarchy:
                    v_out = self.vertical_transformer(enc)
                    h_out = self.horizontal_transformer(enc)
                    window.append((high_pos, x_out, v_in, h_in, v_out, h_out))
                else:
                    window.append((high_pos, x_out, v_in, h_in))
                in_idx += 1
                if inputs.shape[0] > 1:
                    pos += (length - 1) / (inputs.shape[0] - 1)
            if len(window) > 0:
                v = torch.stack([item[2] for item in window])
                h = torch.stack([item[3] for item in window])
                m = torch.tensor([item[0] - block_size // 2 for item in window], device=inputs.device).view(-1, 1, 1)
                scale = self.hf(v, h, m, out_idx)
                x = torch.stack([item[1][out_idx - item[0]] for item in window])
                outx = torch.sum(scale * x, dim=0)
                out[out_idx] = outx
                if self.output_hierarchy:
                    v = torch.stack([item[4][out_idx - item[0]] for item in window])
                    h = torch.stack([item[5][out_idx - item[0]] for item in window])
                    outv = torch.sum(scale * v, dim=0)
                    out_vertical[out_idx] = outv
                    outh = torch.sum(scale * h, dim=0)
                    out_horizontal[out_idx] = outh
            else:
                print("Warning: some sequence information was lost, try increasing block size")
        if self.output_hierarchy:
            return out, (out_vertical, out_horizontal)
        else:
            return out

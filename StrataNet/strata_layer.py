from collections import deque
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x, idx=0):
        pe = torch.zeros(x.shape[0], x.shape[2], device=self.device)
        position = torch.arange(idx, idx + x.shape[0], dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.shape[2], 2, dtype=torch.float32, device=self.device) * -(math.log(10000.0) / x.shape[2]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return x + pe

class HierarchyFunction(nn.Module):
    def __init__(self):
        super(HierarchyFunction, self).__init__()

    def forward(self, x, vertical, horizontal, linspace):
        y = torch.arange(0, x.shape[0], dtype=torch.float32, device=self.device)

        # y = torch.stack([v ** 2 * torch.exp(-()) for n in ])

class StrataLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.1, num_layers=6, output_hierarchy=True):
        super(StrataLayer, self).__init__()
        self.output_hierarchy = output_hierarchy
        # self.input_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
        #     num_layers)
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
        # self.reset_parameters()

    def forward(self, inputs, length, block_size, hierarchy=None):
        if hierarchy is None:
            hierarchy = (torch.zeros_like(inputs), torch.zeros_like(inputs))
        vertical, horizontal = hierarchy
        # if max_length is None:
        #     enc = torch.stack([self.pe(x.repeat(length, 1, 1)) for x in inputs])
        #     out = torch.stack([self.output_transformer(x) for x in enc])
        #     out = 
        # in_lin = torch.linspace(0, length, inputs.shape[0], device=self.device)
        idxs = torch.arange(0, length, dtype=torch.float32, device=self.device)
        window = []
        pos = 0.0
        in_idx = 0
        # dq = deque(maxlen=max_length)
        out = torch.zeros(length, inputs.shape[1], inputs.shape[2], device=self.device)
        if self.output_hierarchy:
            out_vertical = torch.zeros_like(out)
            out_horizontal = torch.zeros_like(out)
        for out_idx in range(length):
            low = out_idx - block_size // 2
            high = out_idx + block_size // 2
            while len(window) > 0 and window[0][0] < low:
                del window[0]
            while in_idx < inputs.shape[0] and high >= pos:
                low_pos = int(pos) - block_size // 2
                high_pos = int(pos) + block_size // 2
                enc = self.pe(inputs[in_idx].repeat(block_size, 1, 1), low_pos)
                x = self.output_transformer(enc)
                if self.output_hierarchy:
                    v = self.vertical_transformer(enc)
                    h = self.horizontal_transformer(enc)
                    window.append((high_pos, x, v, h))
                else:
                    window.append((high_pos, x))
                del enc
                in_idx += 1
                pos += (length - 1) / (inputs.shape[0] - 1)
            # TODO
        # for i in range(inputs.shape[0]):
        #     x = inputs[i]
        #     v = vertical[i]
        #     h = horizontal[i]
        #     enc = self.pe(x.repeat(length, 1, 1))
        #     lin = torch.arange()
        #     if max_length is None:
        #         outx = self.hf(self.output_transformer(enc), v, h)
        #         out += outx
        #         if self.output_hierarchy:
        #             outv = self.hf(self.vertical_transformer(enc), v, h, lin)
        #             outh = self.hf(self.horizontal_transformer(enc), v, h, lin)
        #             out_vertical += outv
        #             out_horizontal += outh
        #     else:
        #         outx = torch.zeros_like(enc)
        #         blockx = enc[]

        if self.output_hierarchy:
            return out, (out_vertical, out_horizontal)
        else:
            return out



    # def reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

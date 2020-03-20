import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        # assert x.dim() == 3
        pe = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        position = torch.arange(0, x.shape[1], device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device) * -(math.log(10000.0) / x.shape[2]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return x + pe

class BaselineTransformer(nn.Module):
    def __init__(self, params):
        super(BaselineTransformer, self).__init__()
        self.params = params
        self.pe = PositionalEncoding()
        self.input_ff = nn.Sequential(
            nn.Linear(params["latent_dim"], params["d_ff"]),
            nn.ReLU(),
            nn.Linear(params["d_ff"], params["d_model"]))
        self.transformer = nn.Transformer(d_model=params["d_model"], nhead=params["n_heads"],
            num_encoder_layers=params["n_layers"], num_decoder_layers=0, dim_feedforward=params["d_ff"],
            dropout=params["dropout"], custom_decoder=nn.Identity())
        self.output_ff = nn.Sequential(
            nn.Linear(params["d_model"], params["d_ff"]),
            nn.ReLU(),
            nn.Linear(params["d_ff"], 128))

    def forward(self, z, steps):
        # assert z.dim() == 2
        # assert type(steps) is int
        x = z.repeat(steps, 1, 1)
        x = self.input_ff(self.pe(x))
        x = self.transformer.encoder(x).transpose(0, 1)
        x = self.output_ff(x)
        return x

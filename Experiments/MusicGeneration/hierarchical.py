import torch
from torch import nn

from StrataNet import StrataLayer


class HierarchicalModel(nn.Module):
    def __init__(self, latent_dims, tier_lengths, num_tiers=3, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=6, block_size=1024):
        super(HierarchicalModel, self).__init__()
        assert len(latent_dims) == num_tiers + 1
        assert len(tier_lengths) == num_tiers - 1
        self.latent_dims = latent_dims
        self.tier_lengths = tier_lengths
        self.num_tiers = num_tiers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.block_size = block_size
        self.tiers = nn.ModuleList(
            StrataLayer(latent_dims[i], latent_dims[i + 1], nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, num_layers=num_layers, output_hierarchy=i < num_tiers - 1,
                output_hierarchy_dim=latent_dims[i + 2] if i < num_tiers - 1 else None)
            for i in range(num_tiers))
        self.output_ff = nn.Linear(latent_dims[-1], 128)

    def forward(self, z, length):
        x = z.unsqueeze(0)
        hp = None
        for i, tier in enumerate(self.tiers):
            if i < self.num_tiers - 1:
                x, hp = tier(x, self.tier_lengths[i], self.block_size, hp)
            else:
                x = tier(x, length, self.block_size, hp)
        x = self.output_ff(x).transpose(0, 1)
        return x

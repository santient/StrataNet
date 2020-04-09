import torch
from torch import nn

from strata_layer import StrataTier, gaussian_kernel


class HierarchicalModel(nn.Module):
    def __init__(self, latent_dims, tier_lengths, num_tiers=3, dropout=0.1, num_layers=6, block_size=50):
        super(HierarchicalModel, self).__init__()
        assert len(latent_dims) == num_tiers + 1
        assert len(tier_lengths) == num_tiers - 1
        self.latent_dims = latent_dims
        self.tier_lengths = tier_lengths
        self.num_tiers = num_tiers
        self.dropout = dropout
        self.num_layers = num_layers
        self.block_size = block_size
        self.kernel = gaussian_kernel
        self.tiers = nn.ModuleList(
            StrataTier(latent_dims[i], latent_dims[i + 1], self.kernel, nhead=latent_dims[i + 1] // 16,
                dim_feedforward=latent_dims[i + 1] * 4, dropout=dropout, num_layers=num_layers, block_size=block_size)
            for i in range(num_tiers))
        self.output_ff = nn.Sequential(
            nn.Linear(latent_dims[-1], latent_dims[-1] * 4),
            nn.ReLU(),
            nn.Linear(latent_dims[-1] * 4, 128))

    def forward(self, z, length):
        x = z.unsqueeze(0)
        for i, tier in enumerate(self.tiers):
            if i < self.num_tiers - 1:
                x = tier(x, self.tier_lengths[i])
            else:
                x = tier(x, length)
        x = self.output_ff(x).transpose(0, 1)
        return x

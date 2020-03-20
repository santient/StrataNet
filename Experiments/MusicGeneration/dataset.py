import glob
import os

import torch
from torch.utils.data import Dataset


class PianorollDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pth")))

    def __len__(self):
        return 13 * len(self.files)

    def __getitem__(self, idx):
        shift = idx % 13 - 6
        x = torch.load(self.files[idx // 13]).type(torch.float32)
        if shift < 0:
            x = torch.cat((x[:, -shift:], torch.zeros_like(x[:, :-shift])), dim=-1)
        if shift > 0:
            x = torch.cat((torch.zeros_like(x[:, -shift:]), x[:, :-shift]), dim=-1)
        return idx, x

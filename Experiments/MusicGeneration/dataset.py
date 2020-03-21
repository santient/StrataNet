import glob
import os

import torch
from torch.utils.data import Dataset


class PianorollDataset(Dataset):
    def __init__(self, root_dir, transpose=False):
        self.root_dir = root_dir
        self.transpose = transpose
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pth")))

    def __len__(self):
        if self.transpose:
            return 13 * len(self.files)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if self.transpose:
            x = torch.load(self.files[idx // 13]).float()
            shift = idx % 13 - 6
            if shift < 0:
                x = torch.cat((x[:, -shift:], torch.zeros_like(x[:, :-shift])), dim=-1)
            if shift > 0:
                x = torch.cat((torch.zeros_like(x[:, -shift:]), x[:, :-shift]), dim=-1)
        else:
            x = torch.load(self.files[idx]).float()
        return idx, x

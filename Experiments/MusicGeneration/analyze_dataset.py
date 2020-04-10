import torch
import tqdm

from dataset import PianorollDataset


dset = PianorollDataset("/results/sbenoit/datasets/lpd_processed/")
results = []
for idx, x in tqdm.tqdm(dset):
    results.append((dset.files[idx], x.shape))
print(sorted(results, key=lambda x: x[1][0]))
print("\n")
print(sorted(results, key=lambda x: -x[1][0]))

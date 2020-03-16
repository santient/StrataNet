import gc
import glob
import os
import random

import numpy
import pypianoroll
import torch
import tqdm


if __name__ == "__main__":
    in_dir = "/media/santiago/BladeProHDD/Datasets/lpd/lpd_cleansed/"
    out_dir = "/media/santiago/BladeProHDD/Datasets/lpd/processed/"  
    paths = glob.glob(os.path.join(in_dir, "**/*.npz"), recursive=True)
    random.shuffle(paths)
    for path in tqdm.tqdm(paths):
        try:
            name = os.path.splitext(os.path.basename(path))[0]
            tracks = pypianoroll.load(path)
            tracks.merge_tracks(track_indices=list(filter(lambda i: not tracks.tracks[i].is_drum, range(len(tracks.tracks)))), mode="any")
            tracks.remove_tracks(range(len(tracks.tracks) - 1))
            tracks.transpose(-6)
            for i in range(13):
                pr = torch.from_numpy(tracks.tracks[-1].pianoroll).type(torch.uint8)
                torch.save(pr, os.path.join(out_dir, name + "[{}].pth".format(i - 6)))
                tracks.transpose(1)
        except Exception:
            print("Exception when processing", path)
        gc.collect()

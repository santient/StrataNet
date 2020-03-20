import os
import json
import torch
from torch.backends import cudnn
import standard_grid
import dataset
import baseline
import train
# import generate


def get_arguments():
    parser = standard_grid.ArgParser()
    # parser.register_parameter("--batch_size", int, 1)
    parser.register_parameter("--epochs", int, 1000)
    parser.register_parameter("--early_stop_threshold", int, 100)
    parser.register_parameter("--model_lr", float, 0.001)
    parser.register_parameter("--latent_lr", float, 0.001)
    parser.register_parameter("--latent_dim", int, 128)
    parser.register_parameter("--d_model", int, 16)
    parser.register_parameter("--n_layers", int, 6)
    parser.register_parameter("--d_ff", int, 512)
    parser.register_parameter("--n_heads", int, 8)
    parser.register_parameter("--dropout", float, 0.1)
    parser.register_parameter("--data_dir", str, "/results/sbenoit/datasets/lpd_processed/")
    args = parser.compile_argparse()
    return args

if __name__ == "__main__":
    params = vars(get_arguments())
    print("Run started with params:", str(params))
    device = torch.device("cuda")
    cudnn.benchmark = True
    out_dir = "output/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    dset = dataset.PianorollDataset(params["data_dir"])
    model = baseline.BaselineTransformer(params).to(device)
    results = train.train(model, dset, params, device, out_dir)
    print("Results:", str(results))
    # generate.generate(model, dset, params, device, out_dir)
    with open(os.path.join(out_dir, "best.txt"), "w") as f:
        f.write(json.dumps(results))
    print("Run finished with params:", str(params))

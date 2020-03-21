import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Parameter, ParameterList, BCEWithLogitsLoss
from torch.optim import Adam


def mean(values):
    return sum(values) / len(values)

def train(model, dset, params, device, out_dir):
    dataloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: (x[0][0], x[0][1].unsqueeze(0)))
    model_opt = Adam(model.parameters(), lr=params["model_lr"])
    latent = ParameterList(Parameter(torch.randn(params["latent_dim"], device=device)) for i in range(len(dset)))
    latent_opts = [Adam([z], lr=params["latent_lr"]) for z in latent]
    criterion = BCEWithLogitsLoss()
    model.train()
    best_epoch = 0
    best_loss = float("inf")
    time_since_improvement = 0
    print("Training model...")
    for epoch in tqdm.tqdm(range(params["epochs"])):
        avg_loss = []
        for idx, x in tqdm.tqdm(dataloader):
            x = x.to(device)
            z = latent[idx].unsqueeze(0)
            # print(x.shape[1])
            gen = model(z, x.shape[1])
            loss = criterion(gen, x)
            avg_loss.append(loss.detach())
            model_opt.zero_grad()
            latent_opts[idx].zero_grad()
            loss.backward()
            model_opt.step()
            latent_opts[idx].step()
        avg_loss = mean(avg_loss)
        if avg_loss < best_loss:
            time_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
            torch.save(latent.state_dict(), os.path.join(out_dir, "latent_best.pth"))
            best_epoch = epoch
            best_loss = avg_loss
        else:
            time_since_improvement += 1
            if time_since_improvement >= params["early_stop_threshold"]:
                break
    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pth")))
    latent.load_state_dict(torch.load(os.path.join(out_dir, "latent_best.pth")))
    return {"best_epoch": best_epoch, "best_loss": best_loss}

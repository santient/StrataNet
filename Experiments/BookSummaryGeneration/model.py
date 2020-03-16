import torch
from torch.nn import Module, AdaptiveLogSoftmaxWithLoss

from StrataNet.vad import StrataVAD


class Model(Module):
    def __init__(self, n_symbols, cutoffs, n_levels, latent_dim,
                 d_model, nhead, num_layers, dim_feedforward, dropout):
        self.n_symbols = n_symbols
        self.cutoffs = cutoffs
        self.vad = StrataVAD(n_levels, latent_dim, d_model, nhead,
                             num_layers, dim_feedforward, dropout)
        # self.log_probs = Sequential([
        #     Linear(self.vad.d_model[-1], self.vad.dim_feedforward[-1]),
        #     ReLU(inplace=True),
        #     Droupout(self.vad.dropout),
        #     Linear(self.vad.dim_feedforward[-1], self.n_symbols)])
        # self.softmax = Softmax()
        self.softmax_loss = AdaptiveLogSoftmaxWithLoss(self.vad.d_model[-1], self.n_symbols, self.cutoffs)

    def forward(self, latent, seq_dims, target=None):
        x = self.vad(latent, seq_dims)
        x = x.view(-1, x.shape[-1])
        if target is not None:
            return self.softmax_loss(x, target.view(-1))
        else:
            probs = self.softmax_loss.log_probs(x)
            return probs.view(-1, latent.shape[0], probs.shape[-1])

    def predict(self, latent, seq_dims):
        x = self.vad(latent, seq_dims)
        x = x.view(-1, x.shape[-1])
        preds = self.softmax_loss.predict(x)
        return preds.view(-1, latent.shape[0]).transpose(0, 1)

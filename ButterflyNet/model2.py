import torch
import torch.nn.functional as F
from torch.nn import Transformer, Module, ModuleList


class ButterflyNet(Module):
    def __init__(self, n_levels, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(ButterflyNet, self).__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be at least 1")
        self.n_levels = n_levels
        self.d_model = self._param_tuple(d_model)
        self.nhead = self._param_tuple(nhead)
        self.num_encoder_layers = self._param_tuple(num_encoder_layers)
        self.num_decoder_layers = self._param_tuple(num_decoder_layers)
        self.dim_feedforward = self._param_tuple(dim_feedforward)
        self.dropout = self._param_tuple(dropout)
        self.transformers = ModuleList([
            Transformer(self.d_model[i], self.nhead[i], self.num_encoder_layers[i],
                        self.num_decoder_layers[i], self.dim_feedforward[i], self.dropout[i])
            for i in range(n_levels)])

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self._param_tuple(tgt)
        src_mask = self._param_tuple(src_mask)
        tgt_mask = self._param_tuple(tgt_mask)
        memory_mask = self._param_tuple(memory_mask)
        src_key_padding_mask = self._param_tuple(src_key_padding_mask)
        tgt_key_padding_mask = self._param_tuple(tgt_key_padding_mask)
        memory_key_padding_mask = self._param_tuple(memory_key_padding_mask)
        x = src
        identity = []
        for i in range(self.n_levels):
            x = x.view(-1, src.shape[-2], x.shape[-1])
            x = self.transformers[i].encoder(
                x, mask=src_mask[i], src_key_padding_mask=src_key_padding_mask[i])
            identity.append(x)
            if i < self.n_levels - 1:
                x = x.view(src.shape[i], -1, x.shape[-1])
                x = self.encoder_rnns[i](x)[0][-1]
        for i in range(self.n_levels):
            if i > 0:
                x = x.view(1, -1, x.shape[-1])
                h = None
                gen = []
                for j in range(tgt[i].shape[0]):
                    x, h = self.decoder_rnns[-i - 1](x, h)
                    gen.append(x[-1])
                x = torch.stack(gen)
            x = self.transformers[-i - 1].decoder(
                tgt[i], x, tgt_mask=tgt_mask[i], memory_mask=memory_mask[i],
                tgt_key_padding_mask=tgt_key_padding_mask[i],
                memory_key_padding_mask=memory_key_padding_mask[i])
            x += identity[-i - 1]
        x = x.view(*tgt[-1].shape)
        return x

    def _param_tuple(self, param):
        if type(param) in (tuple, list):
            if len(param) == self.n_levels:
                return tuple(param)
            else:
                raise ValueError("number of parameters in tuple/list should be n_levels")
        else:
            return (param,) * self.n_levels

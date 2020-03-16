import torch
import torch.nn.functional as F
from torch.nn import Transformer, Linear, Module, ModuleList, Identity


# class VADLatent(Module):
#     def __init__(self, keys=None):
#         self.latent = ParameterDict()
#         if keys is not None:
#             self.init_latent()

class StrataVAD(Module):
    def __init__(self, n_levels, latent_dim, d_model, nhead,
                 num_layers, dim_feedforward, dropout, linears=True):
        super(StrataVAD, self).__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be at least 1")
        self.n_levels = n_levels
        self.latent_dim = latent_dim
        self.d_model = self._param_tuple(d_model)
        if not linears and len(set(self.d_model)) > 1:
            raise ValueError("all d_model must be the same without linears")
        self.nhead = self._param_tuple(nhead)
        self.num_layers = self._param_tuple(num_layers)
        self.dim_feedforward = self._param_tuple(dim_feedforward)
        self.dropout = self._param_tuple(dropout)
        self.transformers = ModuleList([
            # TransformerDecoder(
            #     TransformerDecoderLayer(self.d_model[i], self.nhead[i],
            #                             self.dim_feedforward[i], self.dropout[i]),
            #     self.num_layers[i], LayerNorm(self.d_model[i]))
            Transformer(d_model=self.d_model[i], nhead=self.nhead[i], num_encoder_layers=0,
                        num_decoder_layers=self.num_layers[i], dim_feedforward=self.dim_feedforward[i],
                        dropout=self.dropout[i], custom_encoder=Identity())
            for i in range(self.n_levels)])
        if linears:
            self.linears = ModuleList([
                Linear(self.d_model[i - 1] if i > 0 else self.latent_dim, self.d_model[i])
                for i in range(self.n_levels)])
        else:
            self.linears = None

    def forward(self, latent, seq_dims):
        if len(seq_dims) != self.n_levels:
            raise RuntimeError("number of seq_dims must be n_levels")
        device = latent.device
        batch_dim = latent.shape[0]
        x = latent
        for i in range(self.n_levels):
            x = x.view(-1, x.shape[-1])
            if self.linears is not None:
                x = self.linears[i](x)
            x = x.unsqueeze(0)
            tgt = torch.zeros(seq_dims[i], x.shape[-2], x.shape[-1], device=device)
            # mask = torch.ones(x.shape[-2], seq_dims[i], dtype=torch.bool, device=device)
            x = self.transformers[i].decoder(tgt, x)
        # return x.view(*seq_dims, -1, x.shape[-1])
        return x.view(-1, batch_dim, x.shape[-1]).transpose(0, 1)

    # def forward(self, src, tgt, src_mask=None, tgt_mask=None,
    #             memory_mask=None, src_key_padding_mask=None,
    #             tgt_key_padding_mask=None, memory_key_padding_mask=None):
    #     tgt = self._param_tuple(tgt)
    #     src_mask = self._param_tuple(src_mask)
    #     tgt_mask = self._param_tuple(tgt_mask)
    #     memory_mask = self._param_tuple(memory_mask)
    #     src_key_padding_mask = self._param_tuple(src_key_padding_mask)
    #     tgt_key_padding_mask = self._param_tuple(tgt_key_padding_mask)
    #     memory_key_padding_mask = self._param_tuple(memory_key_padding_mask)
    #     x = src
    #     identity = []
    #     for i in range(self.n_levels):
    #         x = x.view(-1, src.shape[-2], x.shape[-1])
    #         x = self.transformers[i].encoder(
    #             x, mask=src_mask[i], src_key_padding_mask=src_key_padding_mask[i])
    #         identity.append(x)
    #         if i < self.n_levels - 1:
    #             x = x.view(src.shape[i], -1, x.shape[-1])
    #             x = self.encoder_rnns[i](x)[0][-1]
    #     for i in range(self.n_levels):
    #         if i > 0:
    #             x = x.view(1, -1, x.shape[-1])
    #             h = None
    #             gen = []
    #             for j in range(tgt[i].shape[0]):
    #                 x, h = self.decoder_rnns[-i - 1](x, h)
    #                 gen.append(x[-1])
    #             x = torch.stack(gen)
    #         x = self.transformers[-i - 1].decoder(
    #             tgt[i], x, tgt_mask=tgt_mask[i], memory_mask=memory_mask[i],
    #             tgt_key_padding_mask=tgt_key_padding_mask[i],
    #             memory_key_padding_mask=memory_key_padding_mask[i])
    #         x += identity[-i - 1]
    #     x = x.view(*tgt[-1].shape)
    #     return x

    def _param_tuple(self, param):
        if type(param) in (tuple, list):
            if len(param) == self.n_levels:
                return tuple(param)
            else:
                raise ValueError("number of parameters in tuple/list should be n_levels")
        else:
            return (param,) * self.n_levels

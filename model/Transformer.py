import torch
import torch.nn as nn

from model.Decoder import Decoder
from Model.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, d_model, n_head, max_len, ffn_hidden, n_layer, drop_prob):
        super().__init__()
        self.encoder = Encoder(d_model, n_head, max_len, ffn_hidden, enc_voc_size, drop_prob, n_layer)
        self.decoder = Decoder(d_model, n_head, max_len, ffn_hidden, dec_voc_size, drop_prob, n_layer)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

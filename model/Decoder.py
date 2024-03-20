import torch
import torch.nn as nn
from embedding.TransformerEmbedding import TransformerEmbedding
from blocks.DecoderLayer import DecoderLayer



class Decoder(nn.Module):
    def __init__(self,  dec_voc_size,  max_len=1024, d_model, ffn_hidden, n_head, n_layer, drop_prob=0.1):
        super().__init__()
        self.emb = TransformerEmbedding(d_model = d_model, drop_prob = drop_prob, max_len = max_len, vocab_size = dec_voc_size)
        self.layers = nn.Modulelist([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for i in range(n_layer)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output

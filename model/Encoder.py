import torch
import torch.nn as nn
from blocks.EncoderLayer import EncoderLayer
from embedding.TransformerEmbedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, drop_prob, n_layer):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, max_len, enc_voc_size, drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)])
    def foward(self,x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

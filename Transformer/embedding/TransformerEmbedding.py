import torch
import torch.nn as nn

from embedding.PositionEncoding import PositionEncoding
from embedding.TokenEmbedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab, d_model)
        self.pos_emb = PositionEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)

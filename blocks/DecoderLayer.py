import torch
import torch.nn as nn
from layers.MHA import MultiHeadAttention
from layers.MLP import MLP

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p =  drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)

        self.ffn = MLP(d_model, ffn_hidden, drop_prob)
        self.norm3 = nn.Layernorm(d_model)
        self.dropout3 = nn.Dropout(p = drop_prob)

    def foward(self, x, enc, trg_mask, src_mask):
        #self attention
        _x = x
        x = self.self_attention(q = x, k = x, v = x,mask = trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        #enc-dec attention
        if enc_dec is not None:
            _x = x
            x = self.enc_dec_attention(q = x, k = enc, v = enc, mask = src_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x




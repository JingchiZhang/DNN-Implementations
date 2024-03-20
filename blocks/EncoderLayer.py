import torch
import torch.nn as nn
from layers.MLP import MLP
from layers.MHA import MultiheadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model,ffn_ hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = MLP(d_model, ffn_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)

    def foward(self,x,src_mask):
        #x = batch, seq, d_model
        attentionscore = self.attention(x,x,x, mask = src_mask)
        #add and norm
        x = self.dropout1(x)
        x = self.norm1(x+attentionscore)
        #ffn
        _x = x
        x = self.ffn(x)
        #add and norm
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        return x
    

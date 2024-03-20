import torch
import torch.nn as nn

class PostionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.unsqueeze(dim = 1)

        _2i = torch.arange(0, d_model, step = 2).float()

        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))


    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_model, hidden_size, drop_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.relu = nn.Relu()
        self.dropout = nn.Dropout(p = drop_prob)

    def foward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

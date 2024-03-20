import torch
import torch.nn as nn


        
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim = -1)
        
    def foward(self, q, k, v, mask = None):
        #batch_size, n_head, seq_len, d_tensor
        #softmax(q*k.t/sqrt(d_tensor))*v
        batch_size, n_head, seq_len, d_tensor = q.size()
        k_t = k.transpose(2,3)
        qk_score = torch.matmul(q, k_t)/math.sqrt(d_tensor)
        if mask:
            qk_score = qk_score.masked_fill(mask==0, -10000)
        attention_score = torch.matmul(self.softmax(qk_score), v)
        return attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = SacleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def split(self, tensor):
        #tensor size : batch, seq_len, d_model
        # split into : batch_size, n_head, seq_len, d_tensor
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transepose(1,2)
        return tensor
        
    def concat(self, tensor):
        #reverse split
        batchsize, n_head, seq_len, d_tensor = tensor.size()
        d_model = n_head * d_tensor
        tensor.transepose(1,2).view(batchsize, seq_len, d_model)
        return tensor
        
    def forward(self, q,k,v ,mask = None):
        q = self.split(self.w_q(q))
        k = self.split(self.w_k(k))
        v = self.split(self.w_v(v))
        
        attention_score = self.attention(q,k,v, mask = mask)
        attention_score = self.concat(attention_score)
        out = self.proj(attention_score)
        return out
        
        

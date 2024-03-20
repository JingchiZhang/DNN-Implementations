import torch
import torch.nn as nn

#Group Query Attention with KV cache

class GroupQueryAttention(nn.Module):
    def __init__(self, n_q_head, n_kv_head, d_model, max_batch_size = 1024, max_seq_len = 1024):
        super().__init__()
        self.n_q_head = n_q_head
        self.n_kv_head = n_kv_head
        self.n_rep = self.n_q_head // self.n_kv_head
        self.d_model = d_model
        self.head_dim = self.d_model // self.n_q_head
        self.w_q = nn.Linear(self.d_model, self.n_q_head * self.head_dim)
        self.w_k = nn.Linear(self.d_model, self.n_kv_head * self.head_dim)
        self.w_v = nn.Linear(self.d_model, self.n_kv_head * self.head_dim)
        self.w_o = nn.:inear(self.d_model, self.d_model)
        
        self.k_cache = torch.zeors(max_batch_size, max_seq_len, self.n_kv_head,self.head_dim).cuda()
        self.v_cache = torch.zeros(max_batch_size, max_seq_len, self.n_kv_head,self.head_dim).cuda()

        self.softmax = nn.Softmax(dim = -1)

    def repeatkv(self, k, self.n_q_head, self.n_kv_head):
        repeat_time = self.n_q_head // self.n_kv_head
        #k: [batchsize, seqLen, n_kv, head_dim] to
        batchsize, seq_len, n_kv, head_dim = k.size() 
        k = k.reshape(batchsize, seq_len, n_kv, 1, head_dim).expand(batchsize, seq_len, n_kv, repeat_time, head_dim).reshape(batchisize, seq_len, n_kv * repeat_time, head_dim)
        return k

    
    def forward(self, q, k, v, mask, start_pos):
        batchsize, seq_len, _ = q.shape
        q = self.w_q(q)  # batch_size, seq_len, d_model
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(batchsize, seq_len, self.n_q_head, self.head_dim)
        k = k.view(batchsize, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(batchsize, seq_len, self.n_kv_head, self.head_dim)
        
        #cache kv
        self.k_cache[:batchsize, start_pos: start_pos + seq_len] = k
        self.v_cache[:batchsize, start_pos: start_pos + seq_len] = v

        keys = self.k_cache[:batchsize, :start_pos + seq_len] #batchsize, seq_len, n_kv_head, d_model
        values = self.v_cache[:batchsize, : start_pos + seq_len]

        #repeat k and v
        keys = self.repeatkv(k)#batch, seq_len, n_kv*repeat, n_dim
        values = self.repeatkv(v)
        
        #compute attention
        # softmax(mask(q * k.t)/sqrt(d_model))@v
        query = q.transpose(1,2) # b,n,seq,head_d
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        score = torch.matmul(q, keys.transpose(2,3))
        if mask:
            score = score.masked_fill_(0, -1000)
        attention_score = self.softmax(score/sqrt(self.head_dim)) 
        attention_score = torch.matmul(attention_socre, values) # batch, n_q, seq, head_dim

        attention_score = attention_score.transpose(1,2).view(batchsize, seq_len, -1)
        
        out = self.w_o(attention_score) 
        
        return out

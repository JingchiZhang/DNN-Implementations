import torch
import torch.nn as nn

from layers.MLP import MLP


class DenseMOE(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__( )
        self.d_model = d_model
        self.n_experts = n_experts
        
        self.router = nn.Linear(self.d_model, self.n_experts)
        self.experts = nn.ModuleList([MLP(self.d_model) for i in range(self.n_experts)])
        self.top_k = top_k

    def forward(self, x):
        logits = self.router(x) #b,s,n
        top_k_logits, indices = logits.topk(self.top_k, dim = -1) #b,s,k

        zeros = torch.full_like(logits, float('-inf'))

        zeros = zeros.scatter(-1,indices, top_k_logits)

        gating_output = nn.functional.softmax(zeros, -1)

        final_output = torch.zeros(x.size()) # bs,s,d
        flat_x = x.view(-1, x.size(-1)) #bs*s,d
        print("flat_x.shape:", flat_x.size())
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        print("flat_gating_output: ", flat_gating_output.size())
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim = -1)
            print("expert_mask.shape: ", expert_mask.size())
            #print(expert_mask)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]

                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask,i].unsqueeze(1)
                

                weight_output = expert_output * gating_scores

                final_output[expert_mask] += weight_output.squeeze(1)
        return final_output

"""
num_expert = 4
top_k = 2
n_embed = 16
torch.manual_seed(0)
input_x = torch.randn(4,8,n_embed)
densemoe = DenseMOE(n_embed, num_expert, top_k) 
output = densemoe(input_x)
print(input_x.size())
"""
            

        

import torch
import torch.nn as nn

from layers.MLP import MLP


class SwitchTransformer(nn.module):
    def __init__(self, capacity_factor, drop_tokens, is_scale_prob, n_experts,  d_model):
        super().__init()__
        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens
        self.d_model = d_model

        self.experts = nn.Modulelist([MLP(self.d_model, self.d_model * 2,self.d_model) for i in range(self.n_expert)])

        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        batch_size, seq_len, d_model = x.size()
        x = x.view(-1, d_model)

        route_prob = self.softmax(self.switch(x)) # [bs*seq, n_experts]
        route_prob_max, routes = torch.max(route_prob, dim = -1) #[bs*seq, n_expert]

        index_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = torch.zeros(x.shape)

        capacity = x.size(0) / self.n_experts * self.capacity_factor

        counts = x.new_tensor([len(index_list[i]) for i in range(self.n_experts)])

        dropped = []
        if self.drop_tokens:
            for i in range(len(self.n_experts)):
                if (counts[i] <= capacity):
                    continue
                #random
                index_list[i] = index_list[i][torch.randperm(len(index_list[i]))]
                dropped.append(index_list[i][capacity:])
                index_list[i] = index_list[i][:capacity]

        expert_out = [self.experts[i](x[index_list[i],:]) for i in range(self.n_experts)]

        for i in range(self.n_experts):
            final_output[index_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        final_output = final_output * route_prob_max.view(-1,1)

        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output

        
        

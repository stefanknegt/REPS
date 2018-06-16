import torch
from torch import Tensor
from torch.nn import functional as F


class MLPValue(torch.nn.Module):
    def __init__(self, layers, activation=F.relu, eta=10, epsilon=.1, learning_rate=1e-3):
        super(MLPValue, self).__init__()
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.activation = activation

        self.eta = torch.nn.Parameter(Tensor([eta]))

        self.epsilon = epsilon

        self.optimizer_all = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer_eta = torch.optim.Adam([self.eta], lr=learning_rate)

    def forward(self, states):
        out = states
        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out

    def get_value(self, states):
        return self(states)

    def get_weights(self, begin_states, end_states, init_states, rewards, gamma, normalized=True):
        '''
        The weight looks at the importance of the samples and is determined by values/rewards
        '''
        begin_values = self(begin_states)
        end_values = self(end_states)
        init_values = self(init_states)

        bell_error = rewards + gamma * end_values + (1 - gamma) * torch.mean(init_values) - begin_values
        max_val = torch.max(bell_error)
        weights = torch.exp((bell_error - max_val) / self.eta)
        if normalized:
            weights = weights / torch.sum(weights)
        return weights

    def get_mse_loss(self, begin_states, cum_sums):
        begin_values = self(begin_states)
        return torch.nn.functional.mse_loss(begin_values, cum_sums)

    def get_reps_loss(self, begin_states, end_states, init_states, rewards, gamma):
        begin_values = self(begin_states)
        end_values = self(end_states)
        init_values = self(init_states)

        bell_error = rewards + gamma * end_values + (1 - gamma) * torch.mean(init_values) - begin_values
        max_val = torch.max(bell_error)
        weights = torch.exp((bell_error - max_val) / self.eta)

        # Calculate the loss according to the formula
        loss = self.eta * self.epsilon + self.eta * torch.log(torch.mean(weights)) + max_val
        return loss

    def reset(self, weight_range=1e-8, bias_range=1e-8):
        for l in self.layers:
            l.weight.data.uniform_(-weight_range/2, weight_range/2)
            if l.bias is not None:
                l.bias.data.uniform_(-bias_range/2, bias_range/2)

    def save(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)



    # def init_back_prop_step(self, begin_states, cum_sums):
    #     loss = self.get_init_loss(begin_states, cum_sums)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data.item()


    # def back_prop_step(self, begin_states, end_states, rewards):
    #     '''
    #     This function calculates the loss for the value function.
    #     '''
    #     loss = self.get_loss(begin_states, end_states, rewards)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data.item()



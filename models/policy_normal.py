import torch
from models.mlp import MLP
from models.simple import Simple
from torch.nn import functional as F
from torch.distributions import Normal
from torch import Tensor


class PolicyNormal():
    def __init__(self, layers, activation=F.relu):
        self.mu_net = Simple(activation)
        self.sigma_net = Simple(activation)

    def get_action(self, state):
        if state.dim() < 2:
            state.unsqueeze_(0)
        mean = self.get_mu(state)
        std_dev = self.get_sigma(state)
        mean.squeeze()
        std_dev.squeeze()
        m = torch.randn(1) * std_dev + mean
        return Tensor(m.data)

    def get_mu(self, states):
        return self.mu_net.forward(states)

    def get_sigma(self, states):
        return torch.ones(states.shape)*0.1

if __name__ == '__main__':
    test = PolicyNormal([1,3,1])
    input = torch.rand(1)

    a = test.get_action(input)

    print(a)
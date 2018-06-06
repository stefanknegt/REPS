import torch
from models.mlp import MLP
from torch.nn import functional as F
from torch.distributions import Normal
from torch import Tensor


class PolicyNormal():
    def __init__(self, layers, activation=F.relu):
        self.mu_net = MLP(layers, activation)
        self.sigma_net = MLP(layers, activation)

    def get_action(self, state):
        mean = self.mu_net.forward(state)
        std_dev = self.get_sigma(state)

        m = Normal(mean, std_dev)
        return m.sample()

    def parameters(self):
        return [self.mu_net.parameters(), self.sigma_net.parameters()]

    def get_mu(self, states):
        return self.mu_net.forward(states)

    def get_sigma(self, states):
        return torch.ones(states.shape)*0.1

if __name__ == '__main__':
    test = PolicyNormal([1,3,1])
    input = torch.rand(1)

    a = test.get_action(input)

    print(a)
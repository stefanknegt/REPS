import torch
from torch import Tensor
from torch.nn import functional as F

from values.value_model import ValueModel

class MLPValue(ValueModel):
    def __init__(self, layers, activation=F.relu, eta=.1, epsilon=.5, learning_rate=1e-3):
        super(MLPValue, self).__init__()
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.activation = activation

        self.eta = torch.nn.Parameter(Tensor([eta]))

        self.epsilon = epsilon

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def get_value(self, states):
        out = states
        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out

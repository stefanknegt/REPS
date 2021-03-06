import torch
from torch import Tensor
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self, layers, activation=F.relu):
        """
        Creates neural network structure with
        :param layers: list with linear layer sizes (input_size, hid1, hid2 , hid3, ..., output_size)
        """
        super(MLP, self).__init__()
        self.layers = []
        self.activation = activation
        self.eta = torch.nn.Parameter(Tensor([1]))
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        """
        :param x: input features
        :return: output of the network
        """
        "Forward pass (layer + activation)"
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        "Last layer without activation (no output domain restriction)"
        x = self.layers[-1](x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    test = MLP([1,3,6,3,1])
    input = torch.rand(1)

    a = test(input)

    print(a)

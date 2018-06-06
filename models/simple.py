import torch
from torch import Tensor
from torch.nn import functional as F

class Simple(torch.nn.Module):
    def __init__(self, activation=F.tanh):
        """
        Creates neural network structure with
        :param layers: list with linear layer sizes (input_size, hid1, hid2 , hid3, ..., output_size)
        """
        super(Simple, self).__init__()
        self.layers = []
        self.activation = activation
        # for i in range(len(layers)-1):
        #     self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.fc1 = torch.nn.Linear(2, 1, bias=None)
        self.eta = torch.nn.Parameter(Tensor([0.1]))

    def forward(self, x):
        """
        :param x: input features
        :return: output of the network
        """

        # "Forward pass (layer + activation)"
        # for i in range(len(self.layers) - 1):
        #     x = self.activation(self.layers[i](x))
        # "Last layer without activation (no output domain restriction)"
        # x = self.layers[-1](x)
        x = torch.cat((x, x**2), 1)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    test = Simple()
    input = Tensor([[2],[4]])

    a = test(input)

    print(a)

import torch
from torch import Tensor
from torch.nn import functional as F


class Simple(torch.nn.Module):
    def __init__(self, input_size=1, activation=None):
        super(Simple, self).__init__()
        self.layers = []
        self.activation = activation
        self.fc1 = torch.nn.Linear(input_size * 2, 1, bias=None)
        self.eta = torch.nn.Parameter(Tensor([1]))

    def forward(self, x):
        x = torch.cat((x, x**2), 1)
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if __name__ == '__main__':
    test = Simple()
    input = Tensor([[2],[4]])

    a = test(input)

    print(a)

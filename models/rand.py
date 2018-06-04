import random

import torch

class Random(torch.nn.Module):
    def __init__(self, range_min=-1, range_max=1):
        super(Random, self).__init__()
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, x):
        return random.uniform(self.range_min, self.range_max)

    def get_action(self, state):
        return self.forward(state)

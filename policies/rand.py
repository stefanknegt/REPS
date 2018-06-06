import random

import torch

class RandomPolicy(torch.nn.Module):
    def __init__(self, range_min=-1, range_max=1):
        super(RandomPolicy, self).__init__()
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, x):
        return random.uniform(self.range_min, self.range_max)

    def get_action(self, state):
        return self.forward(state)

    def optimize(self, train_dataset, val_dataset, batch_size, learning_rate, verbose=False):
        if verbose: print("hurr durr, I'm a random policy")


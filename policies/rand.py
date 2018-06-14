import random
import torch

class RandomPolicy(torch.nn.Module):
    def __init__(self, range_min=-1, range_max=1, dim=1, is_discrete=False):
        super(RandomPolicy, self).__init__()
        self.range_min = range_min
        self.range_max = range_max
        self.dimensionality = dim
        self.is_discrete = is_discrete

    def forward(self, x):
        if self.is_discrete:
            return torch.randint(self.range_min, self.range_max+1, (self.dimensionality, ))
        else:
            return torch.Tensor([random.uniform(self.range_min, self.range_max) for d in range(self.dimensionality)])

    def get_action(self, state):
        return self.forward(state)

    def optimize(self, max_epochs_opt, train_dataset, val_dataset, batch_size, learning_rate, verbose=False):
        if verbose: print("[policy] hurr durr, I'm a random policy")


import torch
from torch import Tensor
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal

from models.mlp import MLP
from models.simple import Simple
from utils.loss import NormalPolicyLoss_1D

class NormalPolicy():
    def __init__(self, initial_policy, layers, activation=F.relu):
        self.initial_policy = initial_policy
        self.mu_net = Simple(activation)
        self.sigma_net = Simple(activation)

    def get_action(self, state):
        # random action if untrained
        if self.initial_policy is not None:
            return self.initial_policy.get_action(state)
        # sample from normal otherwise
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

    def optimize(self, train_dataset, val_dataset, batch_size, learning_rate, verbose=False):
        # init optimizers
        optimizer_mu = optim.Adagrad(self.mu_net.parameters(), lr=learning_rate)
        optimizer_sigma = optim.Adagrad(self.sigma_net.parameters(), lr=learning_rate)
        # train on batches
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epochs_opt_no_decrease < 5):
            for batch_idx, batch in enumerate(train_dataset):
                optimizer_mu.zero_grad()
                optimizer_sigma.zero_grad()
                # forward pass
                mu = self.mu_net(batch[0])
                sigma = self.get_sigma(batch[0])
                loss = NormalPolicyLoss_1D(mu, sigma, batch[1], batch[2])
                # backpropagate
                loss.backward()
                optimizer_mu.step()
                optimizer_sigma.step()
            # calculate loss on validation data
            mu = self.get_mu(val_dataset[0])
            sigma = self.get_sigma(val_dataset[0])
            cur_loss_opt = NormalPolicyLoss_1D(mu, sigma, val_dataset[1], val_dataset[2])
            # evaluate optimization iteration
            if verbose: print("[policy] epoch:", epoch_opt+1, "| loss:", cur_loss_opt)
            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                epochs_opt_no_decrease = 0
            else:
                epochs_opt_no_decrease += 1
            last_loss_opt = cur_loss_opt
            epoch_opt += 1
        # remove reliance on initial policy
        self.initial_policy = None

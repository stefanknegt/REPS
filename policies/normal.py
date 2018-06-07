import torch
from torch import Tensor
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.data.dataloader import DataLoader

from models.mlp import MLP
from models.simple import Simple
from utils.loss import NormalPolicyLoss_1D

class NormalPolicy(torch.nn.Module):
    def __init__(self, activation=F.relu):
        super(NormalPolicy, self).__init__()
        self.mu1 = torch.nn.Parameter(Tensor([.0]))
        self.mu2 = torch.nn.Parameter(Tensor([.0]))
        self.sigma = torch.nn.Parameter(Tensor([1.]))
        # self.mu_net.fc1.weight.data = Tensor([[0, 0]])

    def forward(self, state):
        return self.get_mu(state) + self.sigma * torch.randn(1)

    def get_mu(self, states):
        return states * self.mu1 + (states**2) * self.mu2

    def get_sigma(self, states):
        return self.sigma

    def get_action(self, state):
        return self.forward(state)
        # commented out for now
        # if state.dim() < 2:
        #     state.unsqueeze_(0)
        # mean = self.get_mu(state)
        # std_dev = self.get_sigma(state)
        # mean.squeeze()
        # std_dev.squeeze()
        # m = torch.randn(1) * std_dev + mean
        # return m.data

    def optimize(self, train_dataset, val_dataset, batch_size, learning_rate, verbose=False):
        # init data loader
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # init optimizers
        optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)
        # train on batches
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                # forward pass
                mu = self.get_mu(batch[0])
                sigma = self.get_sigma(batch[0])
                loss = NormalPolicyLoss_1D(mu, sigma, batch[1], batch[2])
                # backpropagate
                loss.backward()
                optimizer.step()
            # calculate loss on validation data
            mu = self.get_mu(val_dataset[0])
            sigma = self.get_sigma(val_dataset[0])
            cur_loss_opt = NormalPolicyLoss_1D(mu, sigma, val_dataset[1], val_dataset[2])
            # evaluate optimization iteration
            if verbose: print("[policy] epoch:", epoch_opt+1, "| loss:", cur_loss_opt)
            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                epochs_opt_no_decrease = 0
                last_loss_opt = cur_loss_opt
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1
        print("mu1:", self.mu1.data, "mu2:", self.mu2.data, "sigma: ", self.sigma.data)

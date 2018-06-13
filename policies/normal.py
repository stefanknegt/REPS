import torch
from torch import Tensor
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.data.dataloader import DataLoader

from models.mlp import MLP
from models.simple import Simple
from utils.loss import NormalPolicyLoss_1D

class NormalPolicy():
    def __init__(self, layers, sigma):
        self.mu_net = MLP(layers)
        self.sigma = Tensor([sigma])

        # self.mu_net.fc1.weight.data = torch.zeros(self.mu_net.fc1.weight.data.shape)
        # self.mu_net.eta.data = torch.ones(1) * 2

    def get_mu(self, states):
        return self.mu_net.forward(states)

    def get_sigma(self, states):
        return self.sigma

    def get_action(self, state):
        # random action if untrained
        # if self.initial_policy is not None:
        #     return self.initial_policy.get_action(state)
        # sample from normal otherwise
        if state.dim() < 2:
            state.unsqueeze_(0)
        mean = self.get_mu(state)
        std_dev = self.get_sigma(state)
        mean.squeeze()
        std_dev.squeeze()
        m = torch.randn(1) * std_dev + mean
        return m.data

    def optimize(self, train_dataset, val_dataset, batch_size, learning_rate, verbose=False):
        # init data loader
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # init optimizers
        optimizer_mu = optim.Adagrad(self.mu_net.parameters(), lr=learning_rate)
        # train on batches
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(train_data_loader):
                optimizer_mu.zero_grad()

                # forward pass
                mu = self.mu_net(batch[0])
                sigma = self.get_sigma(batch[0])
                loss = NormalPolicyLoss_1D(mu, sigma, batch[1], batch[2])
                # backpropagate
                loss.backward()
                optimizer_mu.step()
            # calculate loss on validation data
            mu = self.get_mu(val_dataset[0])
            sigma = self.get_sigma(val_dataset[0])
            cur_loss_opt = NormalPolicyLoss_1D(mu, sigma, val_dataset[1], val_dataset[2])
            # evaluate optimization iteration
            if verbose: print("[policy] epoch:", epoch_opt+1, "| loss:", cur_loss_opt)
            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                best_model = self.mu_net.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = cur_loss_opt
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1
        self.mu_net.load_state_dict(best_model)
        # if verbose: print("[policy] Thetas:", self.mu_net.fc1.weight.data.data, "sigma: ", self.mu_net.eta.data)


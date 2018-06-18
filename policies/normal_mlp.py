import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.utils.data.dataloader import DataLoader
from utils.data import *


class MLPNormalPolicy(torch.nn.Module):
    def __init__(self, layers, sigma, activation=F.relu, learning_rate=1e-3, act_bound=np.inf, init_value=1e-8):
        super(MLPNormalPolicy, self).__init__()

        self.layers = []
        self.activation = activation
        self.act_bound = act_bound

        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.layers = torch.nn.ModuleList(self.layers)

        self.sigma = torch.nn.Parameter(sigma * torch.ones(layers[-1]))

        #Initialize the adam optimizer and set its learning rate.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Reset mu weights (to make mean around 0 at start) - default set to 1e-8
        self.reset(weight_range=init_value, bias_range=init_value)

    def forward(self, states):
        return self.get_action(states)

    def get_mu(self, states):
        x = states
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        if not np.any(np.isinf(self.act_bound)):
            x = self.act_bound * F.tanh(x)
        return x

    def get_action(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with diagonal covariance.
        '''
        mu = self.get_mu(states)
        distr = MultivariateNormal(mu, torch.diag(F.softplus(self.sigma)))
        return distr.sample()

    def get_action_determ(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with diagonal covariance.
        '''
        mu = self.get_mu(states)
        return mu

    def get_loss(self, begin_states, actions, weights):
        mu = self.get_mu(begin_states)
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        log_likelihood = distr.log_prob(actions)
        loss = - torch.dot(weights.squeeze(), log_likelihood.squeeze())
        return loss

    def reset(self, weight_range=1e-8, bias_range=1e-8):
        for l in self.layers:
            l.weight.data.uniform_(-weight_range/2, weight_range/2)
            if l.bias is not None:
                l.bias.data.zero_()

    def optimize_loss(self, train_dataset, val_dataset, max_epochs, batch_size, verbose=False):
        if batch_size <= 0:
            batch_size = len(train_dataset)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0

        while (epoch_opt < max_epochs) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                prev_states = batch[:][0]
                actions = batch[:][1]
                weights = batch[:][5]

                # back prop steps

                loss = self.get_loss(prev_states, actions, weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # calculate validation loss
            valid_loss = self.get_loss(val_dataset[0], val_dataset[1], val_dataset[5])
            if verbose:
                sys.stdout.write('\r[policy] epoch: %d / %d | loss: %f' % (epoch_opt+1, max_epochs, valid_loss))
                sys.stdout.flush()

            # check if loss is decreasing
            if (last_loss_opt is None) or (valid_loss < last_loss_opt):
                best_model = self.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = valid_loss
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1

        # use best previously found model
        self.load_state_dict(best_model)
        if verbose:
            sys.stdout.write('\r[policy] training complete (%d epochs, %f best loss)' % (epoch_opt, last_loss_opt) + (' ' * (len(str(max_epochs))) * 2 + '\n'))

    def save(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)

    # def back_prop_step(self, begin_states, actions, weights):
    #     '''
    #     This functions calculates the loss for the policy used
    #     '''
    #     loss = self.get_loss(begin_states, actions, weights)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data.item()

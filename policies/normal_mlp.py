import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
import math
import numpy as np
from utils.data import *


class MLPNormalPolicy(torch.nn.Module):
    def __init__(self, layers, sigma, activation=F.relu, learning_rate=1e-3, act_bound=np.inf):
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
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        return distr.sample()

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
                l.bias.data.uniform_(-bias_range/2, bias_range/2)

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




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
import numpy as np

from policies.policy_model import PolicyModel
from utils.data import *


class MLPNormalPolicy(PolicyModel):
    def __init__(self, layers, sigma, activation=F.relu, learning_rate=1e-3):
        super(PolicyModel, self).__init__()

        self.layers = []
        self.activation = activation

        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.layers = torch.nn.ModuleList(self.layers)

        self.sigma = torch.nn.Parameter(sigma * torch.ones(layers[-1]))

        #Initialize the adam optimizer and set its learning rate.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def get_mu(self, states):
        x = states
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x


    def get_action(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with diagonal covariance.
        '''
        mu = self.get_mu(states)
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        return F.tanh(distr.sample())


    def get_loss(self, begin_states, actions, weights):
        mu = self.get_mu(begin_states)
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        log_likelihood = distr.log_prob(actions)
        loss = - torch.dot(weights.squeeze(), log_likelihood.squeeze()) / (torch.sum(weights)) #we want to normalize for weights
        check_values(loss_policy=loss, weights=weights)
        return loss

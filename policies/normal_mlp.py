import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from torch.distributions import MultivariateNormal
from policies.policy_model import PolicyModel
from utils.data import *
from torch import Tensor


class NormalPolicy_MLP(PolicyModel):
    def __init__(self, layers, lr, sigma, activation=F.relu, act_bound=np.inf):
        super(PolicyModel, self).__init__()

        self.layers = []
        self.activation = activation
        self.sigma = torch.nn.Parameter(sigma * torch.ones(layers[-1]))
        self.act_bound = act_bound

        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.layers = torch.nn.ModuleList(self.layers)

        #Initialize the adam optimizer and set its learning rate.
        self.policy_optimizer = torch.optim.SGD(self.parameters(), lr=lr)


    def forward(self, x):
        """
        :param x: input features
        :return: output of the network
        """
        "Forward pass (layer + activation)"
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        "Last layer without activation (no output domain restriction)"
        x = self.layers[-1](x)
        if not np.any(np.isinf(self.act_bound)):
            x = self.act_bound * F.tanh(x)
        return x

    def get_mu(self, states):
        return self.forward(states)

    def get_action(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with diagonal covariance.
        '''
        mu = self.forward(states)
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        return distr.sample()

    def get_loss(self, begin_states, actions, weights):
        mu = self.get_mu(begin_states)
        distr = MultivariateNormal(mu, torch.diag(self.sigma))
        log_likelihood = distr.log_prob(actions)
        # check_values(ll_policy=log_likelihood, ll_term_2=torch.mul(actions - mu, actions - mu) / (2 * (self.sigma ** 2)), ll_term_1=torch.log(self.sigma))

        # print(begin_states.shape, actions.shape, mu.shape, sigma.shape, log_likelihood.shape, weights.shape)
        loss = - torch.dot(weights.squeeze(), log_likelihood.squeeze()) / (torch.sum(weights)) #we want to normalize for weights
        check_values(loss_policy=loss, weights=weights)
        return loss

    def back_prop_step(self, begin_states, actions, weights):
        '''
        This functions calculates the loss for the policy used
        '''
        loss = self.get_loss(begin_states, actions, weights)
        #Take optimizer step
        self.policy_optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.policy_optimizer.step()
        return loss.data.item()
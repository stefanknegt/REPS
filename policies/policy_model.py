import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
import numpy as np
import random
from utils.data import *

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, mu_range, sigma_range, lr):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        #MLP for the action network
        self.policy_hidden_weights = Variable(mu_range / 2 - mu_range * torch.rand(self.hidden_dim, self.state_dim), requires_grad=True)
        self.policy_hidden_bias = Variable(mu_range / 2 - mu_range * torch.rand(self.hidden_dim, 1), requires_grad=True)
        self.policy_mu_weights = Variable(mu_range / 2 - mu_range * torch.rand(self.action_dim, self.hidden_dim), requires_grad=True)
        self.policy_mu_bias = Variable(mu_range / 2 - mu_range * torch.rand(self.action_dim, 1), requires_grad=True)
        self.policy_sigma = Variable(sigma_range * torch.rand(self.action_dim, 1), requires_grad=True)

        #Initialize the adam optimizer and set its learning rate.
        self.policy_optimizer = torch.optim.SGD([self.policy_hidden_weights, self.policy_hidden_bias, self.policy_mu_weights, self.policy_mu_bias, self.policy_sigma], lr = lr)

    def get_action(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with Gaussian noise eps.
        '''
        hidden = torch.relu(torch.matmul(self.policy_hidden_weights, states.transpose(0, 1)) + self.policy_hidden_bias)
        mu = torch.matmul(self.policy_mu_weights, hidden) + self.policy_mu_bias
        sigma = F.softplus(self.policy_sigma)

        eps = Variable(torch.randn(1, mu.size()[1]).float())
        action =  mu + torch.matmul(sigma, eps)
        return action, mu, sigma

    def get_loss(self, begin_states, actions, weights):
        N = len(begin_states)
        _, mu, sigma = self.get_action(begin_states)
        mu = mu.transpose(0, 1)
        check_values(mu=mu,sigma=sigma)

        #TODO: Double check dimensions of squares
        log_likelihood = torch.log(sigma + 1e-6) + torch.mul(actions - mu, actions - mu) / (2 * (sigma ** 2 + 1e-6))
        check_values(ll_policy=log_likelihood, ll_term_2=torch.mul(actions - mu, actions - mu) / (2 * (sigma ** 2 + 1e-6)), ll_term_1=torch.log(sigma + 1e-6))

        # print(begin_states.shape, actions.shape, mu.shape, sigma.shape, log_likelihood.shape, weights.shape)
        loss = torch.sum(torch.mul(weights, log_likelihood) / (torch.sum(weights) + 1)) #we want to normalize for weights
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

    def save(self,path):
        with open(path, 'wb') as f:
            torch.save(self, f)

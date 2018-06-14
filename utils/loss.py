import torch
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np

def log_sum_exp(value):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))

    return m + torch.log(sum_exp)


def bellman_error(eta, vs_curr, vs_next, rewards):
    return (rewards+vs_next-vs_curr)/eta


def REPSLoss(epsilon, eta, vs_curr, vs_next, rewards):
    """

    :param epsilon: epsilon hyperparameter
    :param eta: value of parameter eta
    :param vs_curr: tensor of values of current states
    :param vs_next: tensor of values of next states after performing action
    :param rewards: rewards obtained when performing actions in states vs_curr
    :return: reps loss
    """

    bell_error = bellman_error(eta, vs_curr, vs_next, rewards)
    lsm = log_sum_exp(bell_error)
    num_samples = torch.Tensor([vs_curr.size(0)])

    return eta * (epsilon - torch.log(num_samples) + lsm)


def NormalPolicyLoss(mu, sigma, actions, weights):
    normalizer = torch.sum(torch.log(sigma),dim=1, keepdim=True)/2
    exponent = torch.sum(torch.mul(torch.mul(actions-mu, 1/sigma), actions-mu), dim=1, keepdim=True)/2
    loss = torch.dot(weights.squeeze(), (exponent + normalizer).squeeze()) / torch.sum(weights)
    return loss


def NormalPolicyLoss_1D(mu, sigma, actions, weights):
    normalizer = torch.log(sigma)
    exponent = ((actions-mu)**2)/(2*(sigma**2))
    loss = torch.dot(weights.squeeze(), (exponent + normalizer).squeeze()) / torch.sum(weights)
    return loss


def WeightedLogLikelihoodLoss(dist, actions, weights):
    log_prob = dist.log_prob(actions)
    loss = -1.0 * torch.sum(torch.mul(weights.squeeze(), log_prob.squeeze())) / torch.sum(weights)
    return loss
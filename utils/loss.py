import torch

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

    return epsilon * eta + eta * torch.log(torch.Tensor([vs_curr.size(0)**(-1)])) + eta * lsm

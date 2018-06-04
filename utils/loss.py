import torch

def REPSLoss(epsilon, eta, vs_curr, vs_next, rewards):
    """

    :param epsilon: epsilon hyperparameter
    :param eta: value of parameter eta
    :param vs_curr: tensor of values of current states
    :param vs_next: tensor of values of next states after performing action
    :param rewards: rewards obtained when performing actions in states vs_curr
    :return: reps loss
    """

    #TODO might be numerically unstable (use logsumexp trick)
    return epsilon * eta + eta * torch.log(torch.sum(torch.exp((rewards+vs_next-vs_curr)/eta))/vs_curr.size(0))

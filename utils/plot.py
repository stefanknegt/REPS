"""
Plotting functions
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

def plot_training_results(average_rewards,mu,sigma):
    plt.plot(average_rewards)
    plt.title("Average rewards")
    plt.show()
    plt.plot(mu)
    plt.title("Maximum mu")
    plt.show()
    plt.title("Maximum sigma")
    plt.plot(sigma)
    plt.show()

def plot_results(begin_states, actions, weights):
    """
    Plot the states and actions and use color to indicate the value of the
    weight for that given datapoint.
    """
    begin_states, actions, weights = begin_states.detach().numpy(), actions.detach().numpy(), weights.detach().numpy()
    cmap = matplotlib.cm.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=min(weights), vmax=max(weights))
    colors = [cmap(normalize(value)) for value in weights]
    fig, ax = plt.subplots(figsize=(10,10))
    plt.title('Weight values for begin states and actions')
    ax.set_xlabel('Begin state')
    ax.set_ylabel('Action')
    ax.scatter(begin_states, actions, color=colors)
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    plt.show()

def plot_grid(controller):
    """
    Plot a grid with begin_states, actions and the weights
    """
    begin_states, actions = np.mgrid[-2:2.01:0.1, -2:2.01:0.1]
    begin_states, actions = begin_states.flatten(), actions.flatten()
    end_states, rewards = controller.get_reward(begin_states, actions)
    begin_states, actions, end_states, rewards = Variable(torch.from_numpy(begin_states).float()), Variable(torch.from_numpy(actions).float()), Variable(torch.from_numpy(end_states).float()), Variable(torch.from_numpy(rewards).float())
    weights = controller.get_weights(begin_states, end_states, rewards)
    controller.plot_results(begin_states,actions, weights)

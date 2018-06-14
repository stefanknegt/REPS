import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
import numpy as np
import random

from utils.data import *
from values.value_model import Value
from policies.policy_model import Policy

class Controller(nn.Module):
    def __init__(self, name, reset_perc, lr, hidden_dim, value_range, mu_range, sigma_range, eta, epsilon):
        super(Controller, self).__init__()
        self.env = gym.make(name)
        self.reset_perc = reset_perc
        self.state = self.env.reset()
        self.state_dim = len(self.state)
        self.action_dim = 6
        self.hidden_dim = hidden_dim

        self.policy_model = Policy(self.state_dim, self.action_dim, hidden_dim, mu_range, sigma_range, lr)
        self.value_model = Value(self.state_dim, hidden_dim, value_range, eta, epsilon, lr)

    def get_batch_data(self, batch_size, trajectories):
        '''
        This functions retrieves a batch of data from the environment using the old policy.
        '''
        prev_states = np.zeros((batch_size, self.state_dim))
        new_states = np.zeros((batch_size, self.state_dim))
        actions = np.zeros((batch_size, self.action_dim))
        rewards = np.zeros((batch_size,1))
        for s in range(batch_size):
            prev_state = np.asarray(self.state)
            _, _, action = self.policy_model.get_action(Variable(torch.transpose(torch.tensor([prev_state]).float(),1,0)))
            new_state, reward, done, info = self.env.step(action.detach().numpy().flatten())

            #We save all data in the tables
            prev_states[s,:] = np.asarray(prev_state)
            new_states[s,:] = np.asarray(new_state)
            actions[s,:] = np.asarray(action.detach().numpy().flatten())
            rewards[s,:] = np.asarray(reward)

            if done or random.uniform(0, 1) < trajectories/batch_size:
                self.state = self.env.reset()
            else:
                self.state = new_state
        return prev_states, new_states, rewards, actions

    def train(self, iterations, batch_size, trajectories, epochs, history_size, verbose):
        '''
        For each learning policy, we gather a batch of data and minimize loss for that batch then update value function/policy.
        '''
        data_dict = {}
        iter_rewards = 0
        for iteration in range(iterations):
            self.env.render() #comment this to not render env.

            data_dict[iteration] = {}
            data_dict[iteration]['begin_states'],data_dict[iteration]['end_states'],data_dict[iteration]['rewards'],data_dict[iteration]['actions'] =  self.get_batch_data(batch_size, trajectories)
            check_values(begin=data_dict[iteration]['begin_states'], end=data_dict[iteration]['end_states'], reward=data_dict[iteration]['rewards'], action=data_dict[iteration]['actions'])
            iter_rewards += sum(data_dict[iteration]['rewards']) / len(data_dict[iteration]['rewards'])
            begin_states, end_states, rewards, actions = data_dict[iteration]['begin_states'],data_dict[iteration]['end_states'],data_dict[iteration]['rewards'],data_dict[iteration]['actions']

            for i in range(iteration-history_size,iteration):
                if i in data_dict.keys():
                    begin_states = np.concatenate((begin_states,data_dict[i]['begin_states']),axis=0)
                    end_states = np.concatenate((end_states,data_dict[i]['end_states']),axis=0)
                    rewards = np.concatenate((rewards,data_dict[i]['rewards']),axis=0)
                    actions = np.concatenate((actions,data_dict[i]['actions']),axis=0)

            begin_states, end_states, rewards, actions = Variable(torch.from_numpy(begin_states.T).float()), Variable(torch.from_numpy(end_states.T).float()), Variable(torch.from_numpy(rewards.T).float()), Variable(torch.from_numpy(actions.T).float())

            #Now we iteratively update the parameters to reduce the loss:
            iter_val_loss = 0
            for epoch in range(epochs):
                epoch_loss = self.value_model.back_prop_step(begin_states, end_states, rewards)
                iter_val_loss += epoch_loss

            #Now we get the weights to update our policy:
            iter_pol_loss = 0
            weights = self.value_model.get_weights(begin_states, end_states, rewards)
            check_values(weights=weights)

            for epoch in range(epochs):
                epoch_loss = self.policy_model.back_prop_step(begin_states, actions, weights)
                iter_pol_loss += epoch_loss

            if iteration % 10 == 0:
                print("Average reward for iteration: ", iteration, iter_rewards / 10)
                iter_rewards = 0

model = Controller(name='HalfCheetah-v2', reset_perc=0.01, lr=0.001, hidden_dim=20, value_range=1, mu_range=0.4, sigma_range=2, eta=0.1, epsilon=0.5)
model.train(iterations=1000, batch_size=200, trajectories=5, epochs=5, history_size=5, verbose=1)

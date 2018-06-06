import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from LQR_class import LQR

import matplotlib
import matplotlib.pyplot as plt

class LQR_REPS_model:
    '''
    This is the REPS model for LQR, it uses a simple model to approximate the value function using a linear combination of theta1/2.
    '''
    def __init__(self, lr):
        self.env = LQR(-1, 1)

        #These are the trainable weights for the torch value model.
        self.theta1 = Variable(torch.tensor([0.00]), requires_grad=True)
        self.theta2 = Variable(torch.tensor([0.00]), requires_grad=True)
        self.eta = Variable(torch.tensor([0.10]), requires_grad=True)

        #Epsilon is the maximum KL divergence term, which is supposed to work well at 0.1.
        self.epsilon = 0.1

        #These are the trainable weights for the torch value model.
        self.mu1 = Variable(torch.tensor([0.00]), requires_grad=True)
        self.mu2 = Variable(torch.tensor([0.00]), requires_grad=True)
        self.sigma = Variable(torch.tensor([1.00]), requires_grad=True)

        #Initialize the adam optimizer and set its learning rate.
        self.value_optimizer = torch.optim.SGD([self.theta1, self.theta2, self.eta], lr = lr)
        self.policy_optimizer = torch.optim.SGD([self.mu1, self.mu2, self.sigma], lr = lr)

    def get_value(self, state):
        #V(S) is a simple quadratic function of the state
        value = self.theta1 * state + self.theta2 * (state ** 2)
        return value

    def get_random_action(self):
        #Action is now random, but has to be trained in the future.
        action = random.uniform(-2, 2)
        return action

    def get_action(self, state):
        #We determine a mu and sigma
        N = len(state)
        mu = self.mu1 * state + self.mu2 * (state ** 2)
        eps = Variable(torch.randn(N)).float()
        return mu + eps * self.sigma

    def get_weights(self, begin_states, end_states, rewards):
        #the weight looks at the importance of the samples:
        begin_values = self.get_value(begin_states)
        end_values = self.get_value(end_states)
        weights = torch.exp((rewards - begin_values + end_values) / self.eta)
        return weights

    def plot_results(self, begin_states, actions, weights):
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

    def plot_grid_results(self):
        """
        Plot a grid with begin_states, actions and the weights
        """

        begin_states, actions = np.mgrid[-2:2.01:0.1, -2:2.01:0.1]
        begin_states, actions = begin_states.flatten(), actions.flatten()
        end_states, rewards = self.env.get_reward(begin_states, actions)
        begin_states, actions, end_states, rewards = Variable(torch.from_numpy(begin_states).float()), Variable(torch.from_numpy(actions).float()), Variable(torch.from_numpy(end_states).float()), Variable(torch.from_numpy(rewards).float())
        weights = self.get_weights(begin_states, end_states, rewards)
        self.plot_results(begin_states,actions, weights)

    def value_back_prop_step(self, begin_states, end_states, rewards):
        N = len(begin_states)
        #Here we determine the value functions before taking gradient.
        begin_values = self.get_value(begin_states)
        end_values = self.get_value(end_states)

        #Calculate the loss according to the formula.
        loss = self.eta * self.epsilon +  self.eta * torch.log(torch.sum(torch.exp((rewards - begin_values + end_values) / self.eta) / N))

        #Take optimizer step
        self.value_optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.value_optimizer.step()
        return loss.data.item()

    def policy_back_prop_step(self, begin_states, actions, weights):
        N = len(begin_states)

        #Calculate the loss according to the formula.
        mu = self.mu1 * begin_states + self.mu2 * (begin_states ** 2)
        log_likelihood = torch.log(self.sigma + 1e-6) + (actions - mu) ** 2 / (2 * (self.sigma ** 2))
        loss = torch.sum(torch.mul(weights, log_likelihood)) / sum(weights) #we want to normalize for weights

        #Take optimizer step
        self.policy_optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.policy_optimizer.step()
        return loss.data.item()

    def get_batch_data(self, batch_size):
        batch_data = []
        for s in range(batch_size):
            #We save all data in dicts
            data = {}
            data['prev_state'] = self.env.state

            #This is a random action:
            #data['action'] = self.get_random_action()

            #This is a Normal distributed action, which can be learned!
            data['action'] = self.get_action(Variable(torch.tensor([data['prev_state']]).float())).item()

            data['new_state'], data['reward'] = self.env.step(data['action'])
            #Put the dict in a list:
            batch_data.append(data)
        return batch_data

    def train(self, batches, batch_size, steps_per_batch):
        #for each learning policy, we gather a batch of data and minimize loss for that batch then update policy:
        for batch in range(batches):
            #Here we get the data from the env and
            batch_data_LD = self.get_batch_data(batch_size) #list of dicts
            batch_data_DL = dict(zip(batch_data_LD[0],zip(*[d.values() for d in batch_data_LD]))) #dict of lists

            begin_states, end_states, rewards, actions = np.asarray(batch_data_DL['prev_state']), np.asarray(batch_data_DL['new_state']), np.asarray(batch_data_DL['reward']), np.asarray(batch_data_DL['action'])

            print("Average rewards are now: ", sum(rewards) / len(rewards))

            begin_states, end_states, rewards, actions = Variable(torch.from_numpy(begin_states).float()), Variable(torch.from_numpy(end_states).float()), Variable(torch.from_numpy(rewards).float()), Variable(torch.from_numpy(actions).float())

            #Now we iteratively update the parameters to reduce the loss:
            batch_loss = 0
            for step in range(steps_per_batch):
                step_loss = self.value_back_prop_step(begin_states, end_states, rewards)
                print("Value loss -- Step, step_loss: ", step, step_loss)
                batch_loss += step_loss

            print("Value loss -- Batch, avg_batch_loss: ", batch, batch_loss / steps_per_batch)

            #Now we get the weights to update our policy:
            weights = self.get_weights(begin_states, end_states, rewards)
            for step in range(steps_per_batch):
                step_loss = self.policy_back_prop_step(begin_states, actions, weights)
                print("Mu1, mu2, sigma: ", self.mu1.item(), self.mu2.item(), self.sigma.item())
                print("Policy loss -- Step, step_loss: ", step, step_loss)

        self.plot_grid_results()


model = LQR_REPS_model(lr=0.01)
model.train(batches = 10, batch_size = 1000, steps_per_batch = 10)

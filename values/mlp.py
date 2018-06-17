import torch
import sys
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from torch.nn import functional as F

import numpy as np
from scipy.optimize import minimize


class MLPValue(torch.nn.Module):
    def __init__(self, layers, activation=F.relu, eta=10, epsilon=.1, learning_rate=1e-3):
        super(MLPValue, self).__init__()
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.activation = activation

        self.eta = torch.nn.Parameter(Tensor([eta]))

        self.epsilon = epsilon

        self.optimizer_all = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer_eta = torch.optim.Adam([self.eta], lr=learning_rate*100)


    def forward(self, states):
        out = states
        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out


    def get_value(self, states):
        return self(states)


    def get_weights(self, begin_states, end_states, init_states, rewards, gamma, normalized=True):
        '''
        The weight looks at the importance of the samples and is determined by values/rewards
        '''
        begin_values = self(begin_states)
        end_values = self(end_states)
        init_values = self(init_states)

        bell_error = rewards + gamma * end_values + (1 - gamma) * torch.mean(init_values) - begin_values
        max_val = torch.max(bell_error)
        weights = torch.exp((bell_error - max_val) / self.eta)
        if normalized:
            weights = weights / torch.sum(weights)
        return weights


    def mse_loss(self, begin_states, cum_sums):
        begin_values = self(begin_states)
        return F.mse_loss(begin_values, cum_sums)


    def reps_loss(self, begin_states, end_states, init_states, rewards, gamma):
        begin_values = self(begin_states)
        end_values = self(end_states)
        init_values = self(init_states)

        bell_error = rewards + gamma * end_values + (1 - gamma) * torch.mean(init_values) - begin_values
        max_val = torch.max(bell_error)
        weights = torch.exp((bell_error - max_val) / self.eta)

        # Calculate the loss according to the formula
        loss = self.eta * self.epsilon + self.eta * torch.log(torch.mean(weights)) + max_val
        return loss


    def reset(self, weight_range=1e-8, bias_range=1e-8):
        for l in self.layers:
            l.weight.data.uniform_(-weight_range/2, weight_range/2)
            if l.bias is not None:
                l.bias.data.zero_()


    def optimize_loss(self, train_dataset, val_dataset, loss_type, optimizer, max_epochs, batch_size, init_states=None, gamma=0, verbose=False):
        if batch_size <= 0:
            batch_size = len(train_dataset)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0

        while (epoch_opt < max_epochs) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                prev_states = batch[:][0]
                actions = batch[:][1]
                rewards = batch[:][2]
                new_states = batch[:][3]
                cum_sums = batch[:][4]
                weights = batch[:][5]

                # back prop steps
                if loss_type == self.mse_loss:
                    loss = self.mse_loss(prev_states, cum_sums)
                else:
                    loss = self.reps_loss(prev_states, new_states, init_states, rewards, gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calculate validation loss
            if loss_type == self.mse_loss:
                valid_loss = self.mse_loss(val_dataset[0], val_dataset[4])
            else:
                valid_loss = self.reps_loss(val_dataset[0], val_dataset[3], init_states, val_dataset[2], gamma)
            if verbose:
                sys.stdout.write('\r[valid] epoch: %d / %d | loss: %f' % (epoch_opt+1, max_epochs, valid_loss))
                sys.stdout.flush()

            # check if loss is decreasing
            if (last_loss_opt is None) or (valid_loss < last_loss_opt):
                best_model = self.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = valid_loss
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1

        # use best previously found model
        self.load_state_dict(best_model)
        if verbose:
            sys.stdout.write('\r[valid] training complete (%d epochs, %f best loss)' % (epoch_opt, last_loss_opt) + (' ' * (len(str(max_epochs))) * 2 + '\n'))


    def save(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)



    # def init_back_prop_step(self, begin_states, cum_sums):
    #     loss = self.get_init_loss(begin_states, cum_sums)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data.item()


    # def back_prop_step(self, begin_states, end_states, rewards):
    #     '''
    #     This function calculates the loss for the value function.
    #     '''
    #     loss = self.get_loss(begin_states, end_states, rewards)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.data.item()



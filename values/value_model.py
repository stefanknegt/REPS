import torch

from utils.data import *

class ValueModel(torch.nn.Module):
    def __init__(self):
        super(ValueModel, self).__init__()


    def forward(self, states):
        return self.get_value(states)


    def get_value(self, states):
        '''
        The value V(S) is a MLP function of the state with relu activation on the hidden layer
        '''
        return None


    def get_weights(self, begin_states, end_states, rewards):
        '''
        The weight looks at the importance of the samples and is determined by values/rewards
        '''
        begin_values = self.get_value(begin_states)
        end_values = self.get_value(end_states)

        exp_values = rewards - begin_values + end_values
        max_val = torch.max(exp_values)

        weights = torch.exp(exp_values - max_val) / self.eta
        return weights


    def get_loss(self, begin_states, end_states, rewards):
        N = len(begin_states)
        begin_values = self(begin_states)
        end_values = self(end_states)
        check_values(begin_from_value=begin_values,end_from_value=end_values)
        # Calculate the loss according to the formula.
        loss = self.eta * self.epsilon + self.eta * logsumexponent((rewards - begin_values + end_values) / self.eta, N) #torch.log(torch.sum(torch.exp((rewards - begin_values + end_values) / self.eta) / N))
        check_values(loss_value=loss)
        return loss


    def back_prop_step(self, begin_states, end_states, rewards):
        '''
        This function calculates the loss for the value function.
        '''
        loss = self.get_loss(begin_states, end_states, rewards)
        #Take optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


    def save(self,path):
        with open(path, 'wb') as f:
            torch.save(self, f)

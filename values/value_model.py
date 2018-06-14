import torch

class ValueModel(torch.nn.Module):
    def __init__(self):
        super(ValueModel, self).__init__()


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
        return None

    def back_prop_step(self, begin_states, end_states, rewards):
        '''
        This function calculates the loss for the value function.
        '''
        return None

    def save(self,path):
        with open(path, 'wb') as f:
            torch.save(self, f)

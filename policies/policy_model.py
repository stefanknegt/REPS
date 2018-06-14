import torch

class PolicyModel(torch.nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()

    def get_action(self, states):
        '''
        Action is a random multivariate Gaussian determined by an MLP with Gaussian noise eps.
        '''
        return None, None, None

    def get_loss(self, begin_states, actions, weights):
        return None

    def back_prop_step(self, begin_states, actions, weights):
        '''
        This functions calculates the loss for the policy used
        '''
        return None

    def save(self,path):
        with open(path, 'wb') as f:
            torch.save(self, f)

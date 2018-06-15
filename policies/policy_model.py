import torch

class PolicyModel(torch.nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()


    def forward(self, states):
        return self.get_action(states)


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
        loss = self.get_loss(begin_states, actions, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()


    def save(self,path):
        with open(path, 'wb') as f:
            torch.save(self, f)

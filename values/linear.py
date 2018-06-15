import torch
from torch.autograd import Variable

from values.value_model import ValueModel

class LinearValue(ValueModel):
    def __init__(self, state_dim, hidden_dim, value_range, eta, epsilon, lr):
        super(LinearValue, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        #MLP for the value network
        self.value_hidden_weights = Variable(value_range / 2 - value_range * torch.rand(self.hidden_dim, self.state_dim), requires_grad=True)
        self.value_hidden_bias = Variable(value_range / 2 - value_range * torch.rand(self.hidden_dim, 1), requires_grad=True)
        self.value_out_weights = Variable(value_range / 2 - value_range * torch.rand(1, self.hidden_dim), requires_grad=True)
        self.value_out_bias = Variable(value_range / 2 - value_range * torch.rand(1, 1), requires_grad=True)
        self.eta = Variable(torch.tensor([eta]), requires_grad=True)

        #Epsilon is the maximum KL divergence term, which is supposed to work well at 0.1.
        self.epsilon = epsilon

        self.optimizer = torch.optim.SGD([self.value_hidden_weights, self.value_hidden_bias, self.value_out_weights, self.value_out_bias, self.eta], lr = lr)


    def get_value(self, states):
        '''
        The value V(S) is a MLP function of the state with relu activation on the hidden layer
        '''
        hidden = torch.relu(torch.matmul(self.value_hidden_weights, states.transpose(0,1)) + self.value_hidden_bias)
        return torch.matmul(self.value_out_weights, hidden) + self.value_out_bias

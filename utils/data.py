import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys

class SARSDataset(Dataset):
    """
    Container class for state, action, reward, new state observations.
    """

    def __init__(self, sars_data=None, cuda=False):
        """

        :param sars_data: [{'prev_state': array, 'action': array, 'reward': array, 'new_state': array, 'cum_sum': array}, ... ]
        """
        self.prev_states = self._dict_to_tensor(sars_data, 'prev_state', cuda) if sars_data is not None else None
        self.actions = self._dict_to_tensor(sars_data, 'action', cuda) if sars_data is not None else None
        self.rewards = self._dict_to_tensor(sars_data, 'reward', cuda) if sars_data is not None else None
        self.new_states = self._dict_to_tensor(sars_data, 'new_state', cuda) if sars_data is not None else None
        self.cum_sums = self._dict_to_tensor(sars_data, 'cum_sum', cuda) if sars_data is not None else None

    def __len__(self):
        return len(self.prev_states) if self.prev_states is not None else 0

    def __getitem__(self, idx):
        return self.prev_states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.cum_sums[idx]

    def _dict_to_tensor(self, sars_data, key, cuda=False):

        if len(sars_data) < 1:
            if cuda:
                return torch.Tensor([]).cuda()
            else:
                return torch.Tensor([])

        # get data dimensionality
        dim = None
        if isinstance(sars_data[0][key], (float, np.float)):
            dim = 1
        elif isinstance(sars_data[0][key], np.ndarray):
            # check for numpy scalars
            dim = sars_data[0][key].shape[0] if len(sars_data[0][key].shape) > 0 else 1
        else:
            dim = len(sars_data[0][key])
        # init result matrix of appropriate size
        res = torch.zeros((len(sars_data), dim))
        if cuda:
            res = torch.zeros((len(sars_data), dim)).cuda()
        for sars_idx, sars in enumerate(sars_data):
            # check for scalars
            sars_datum = sars[key] if dim > 1 else [float(sars[key])]
            # cast to Tensor
            res[sars_idx] = torch.Tensor(sars_datum)
            if cuda:
                res[sars_idx] = torch.Tensor(sars_datum).cuda()
        return res

    def append(self, sars_data):
        """
        Append new observations to the dataset.
        """
        # if dataset contains no data yet
        if self.prev_states is None:
            self.__init__(sars_data)
        # append otherwise
        else:
            self.prev_states = torch.cat((self.prev_states, self._dict_to_tensor(sars_data, 'prev_state')), dim=0)
            self.actions = torch.cat((self.actions, self._dict_to_tensor(sars_data, 'action')), dim=0)
            self.rewards = torch.cat((self.rewards, self._dict_to_tensor(sars_data, 'reward')), dim=0)
            self.new_states = torch.cat((self.new_states, self._dict_to_tensor(sars_data, 'new_state')), dim=0)
            self.cum_sums = torch.cat((self.cum_sums, self._dict_to_tensor(sars_data, 'cum_sum')), dim=0)

    def concatenate(self, sars_dataset):
        if self.prev_states is None:
            self.prev_states = sars_dataset.prev_states
            self.actions = sars_dataset.actions
            self.rewards = sars_dataset.rewards
            self.new_states = sars_dataset.new_states
            self.cum_sums = sars_dataset.cum_sums
        else:
            self.prev_states = torch.cat((self.prev_states, sars_dataset.prev_states), dim=0)
            self.actions = torch.cat((self.actions, sars_dataset.actions), dim=0)
            self.rewards = torch.cat((self.rewards, sars_dataset.rewards), dim=0)
            self.new_states = torch.cat((self.new_states, sars_dataset.new_states), dim=0)
            self.cum_sums = torch.cat((self.cum_sums, sars_dataset.cum_sums), dim=0)


def logsumexponent(term,N):
    '''
    This is a trick to avoid overflow in the log.
    '''
    max_val = torch.max(term)
    term_corrected = term - max_val
    sumOfExp = torch.sum(torch.exp(term_corrected) / N)
    return max_val + torch.log(sumOfExp)

def check_values(*args,**kwargs):
    "Function to check where NaN values arise"
    nan=False
    for k,v in kwargs.items():
        if type(v) is np.ndarray:
            if np.isnan(v.any()):
                print("NAN DETECTED FOR:",k)
                nan=True
        else:
            if torch.isnan(torch.sum(v)):
                print("NAN DETECTED FOR:",k)
                nan=True

    if nan == True:
        for k,v in kwargs.items():
            print(k,v)
        sys.exit()
    return 0

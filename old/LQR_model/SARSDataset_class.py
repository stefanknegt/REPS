import numpy as np
from torch.utils.data import Dataset

class SARSDataset(Dataset):
    """
    Container class for state, action, reward, new state observations.
    """

    def __init__(self, sars_data):
        '''
        Input:
            sars_data: [{'prev_state': array, 'action': array, 'reward': array, 'new_state'}, ... ]
        '''
        self.sars_data = self._dict_to_nparray(sars_data)

    def __len__(self):
        return len(self.sars_data)

    def __getitem__(self, idx):
        return self.sars_data[idx]

    def _dict_to_nparray(self, sars_data):
        res = np.zeros((len(sars_data), 4))
        for sars_idx, sars in enumerate(sars_data):
            cur_sars = np.array([sars['prev_state'], sars['action'], sars['reward'], sars['new_state']])
            res[sars_idx] = cur_sars
        return res


def main():
    import LQR_class
    batch_data = LQR_class.main()

    sars_dataset = SARSDataset(batch_data)

    print(sars_dataset[0:3])
    return sars_dataset

if __name__ == '__main__':
    main()
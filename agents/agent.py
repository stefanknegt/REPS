import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.sars import SARSDataset
from utils.loss import REPSLoss

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

class Agent:
    """
    Superclass for RL agents.
    """

    def __init__(self, environment, policy_model, value_model):
        """
        Constructor of agent.

        :param environment: environment object with step and reset functions
        :param policy model (nn.Module): optimizable policy estimator
        :param value model (nn.Module): with optimizable value estimator
        """
        self.environment = environment
        self.policy_model = policy_model
        self.value_model = value_model

        self.observations = SARSDataset()

    def get_action(self, state):
        """
        Get action given current state according to the current policy.

        :param state: appropriate representation of the current state
        :return appriate action representation
        """
        return self.policy_model.get_action(state)

    def explore(self, timesteps):
        """
        Explore the environment for t timesteps.

        :param timesteps (int): number of timesteps to explore
        """
        # initialize exploration
        new_observations = []
        cur_state = self.environment.state
        for t in range(timesteps):
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            new_state, new_reward = self.environment.step(cur_action)
            # save new observation
            new_observations.append({
                    'prev_state': cur_state,
                    'action': cur_action,
                    'reward': new_reward,
                    'new_state':new_state})
            # iterate
            cur_state = new_state

        self.observations.append(new_observations)
        return new_observations

    def improve_policy(self):
        pass

    def improve_values(self):
        # init hyperparameters
        timesteps = 4096
        batch_size = 8
        learning_rate = 1e-3
        epsilon = .1
        eta = 1
        # explore
        observations = SARSDataset(self.explore(timesteps))
        data_loader = DataLoader(observations, batch_size=batch_size, shuffle=True, num_workers=4)
        # train on batches
        optimizer = optim.Adagrad(self.value_model.parameters(), lr=learning_rate)
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            # forward pass
            prev_states = batch[:,0].view(batch_size, 1)
            value_predictions = self.value_model(prev_states)
            # calculate loss
            rewards = batch[:,2].view(batch_size, 1)
            new_states = batch[:,3].view(batch_size, 1)
            loss = REPSLoss(epsilon, eta, value_predictions, new_states, rewards)
            print("current loss:", loss)
            # backpropagate
            loss.backward()
            optimizer.step()

def main():
    from environments.lqr import LQR
    from models.rand import Random
    from models.mlp import MLP

    environment = LQR(-10, 10)
    policy_model = Random()
    value_model = MLP([1, 16, 1])

    agent = Agent(environment, policy_model, value_model)
    agent.improve_values()

if __name__ == '__main__':
    main()
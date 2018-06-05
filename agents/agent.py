import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.sars import SARSDataset
from utils.loss import REPSLoss

import torch
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

    def calc_loss(self, batch, batch_size, epsilon):
        # forward pass
        prev_states = batch[:,0].view(batch_size, 1)
        new_states = batch[:,3].view(batch_size, 1)
        prev_value_predictions = self.value_model(prev_states)
        new_value_predictions = self.value_model(new_states)
        # calculate loss
        rewards = batch[:,2].view(batch_size, 1)
        return REPSLoss(epsilon, self.value_model.eta, prev_value_predictions, new_value_predictions, rewards)

    def calc_weights(self, prev_states, new_states, rewards):
        return torch.exp((rewards - prev_states + new_states)/self.value_model.eta)

    def improve_policy(self):
        pass

    def improve_values(self, max_epochs_exp=10, max_epochs_opt=200, timesteps=256, batch_size=256, learning_rate=1e-1, epsilon=0.1):
        # sanity check
        batch_size = min(timesteps, batch_size)
        # init optimizer
        optimizer = optim.Adagrad(self.value_model.parameters(), lr=learning_rate)
        # exploration loop
        last_loss_exp = None
        epochs_exp_no_decrease = 0
        epoch_exp = 0
        while (epoch_exp < max_epochs_exp) and (epochs_exp_no_decrease < 5):
            # reset environment
            self.environment.reset()
            # explore
            observations = SARSDataset(self.explore(timesteps))
            data_loader = DataLoader(observations, batch_size=batch_size, shuffle=True, num_workers=4)
            # train on batches
            last_loss_opt = None
            epochs_opt_no_decrease = 0
            epoch_opt = 0
            while (epoch_opt < max_epochs_opt) and (epochs_opt_no_decrease < 5):
                for batch_idx, batch in enumerate(data_loader):
                    optimizer.zero_grad()
                    loss = self.calc_loss(batch, batch_size, epsilon)
                    # backpropagate
                    loss.backward()
                    optimizer.step()
                epoch_opt += 1
                # evaluate optimization iteration
                cur_loss_opt = self.calc_loss(observations[:], len(observations), epsilon)
                print(epoch_exp, "-", epoch_opt, "epoch, loss:", cur_loss_opt)
                if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                    epochs_opt_no_decrease = 0
                else:
                    epochs_opt_no_decrease += 1
                last_loss_opt = cur_loss_opt
                epoch_opt += 1
            # evaluate exploration iteration
            if (last_loss_exp is None) or (last_loss_opt < last_loss_exp):
                epochs_exp_no_decrease = 0
            else:
                epochs_exp_no_decrease += 1
            last_loss_exp = last_loss_opt
            epoch_exp += 1

def main():
    from environments.lqr import LQR
    from models.rand import Random
    from models.simple import Simple

    import random
    random.seed(42)

    environment = LQR(-2, 2)
    policy_model = Random(-2, 2)
    value_model = Simple()

    agent = Agent(environment, policy_model, value_model)
    agent.improve_values()
    print([ p for p in agent.value_model.parameters()])

if __name__ == '__main__':
    main()
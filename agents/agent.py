import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data import SARSDataset
from utils.loss import REPSLoss
from utils.loss import NormalPolicyLoss_1D

from models.policy_normal import PolicyNormal

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

class Agent:
    """
    Superclass for RL agents.
    """

    def __init__(self, environment, policy_model, value_model, verbose=False):
        """
        Constructor of agent.

        :param environment: environment object with step and reset functions
        :param policy model (nn.Module): optimizable policy estimator
        :param value model (nn.Module): with optimizable value estimator
        :param verbose (bool): verbosity of the agent
        """
        self.environment = environment
        self.policy_model = policy_model
        self.value_model = value_model

        self.observations = SARSDataset()

        self.verbose = verbose

    def get_action(self, state):
        """
        Get action given current state according to the current policy.

        :param state: appropriate representation of the current state
        :return appriate action representation
        """
        return self.policy_model.get_action(state)

    def explore(self, episodes, timesteps):
        """
        Explore the environment for t timesteps.

        :param episodes (int): number of episodes to explore
        :param timesteps (int): number of timesteps per episode
        :return newly collected observations
        """
        # initialize exploration
        if self.verbose: print("exploring", episodes, "with", timesteps, "timesteps each")
        new_observations = []
        cur_state = self.environment.state
        for t in range(episodes*timesteps):
            # reset environment
            if t % timesteps == 0:
                cur_state = self.environment.reset()
            # perform action according to policy
            cur_action = self.get_action(Tensor([cur_state]))
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
        if self.verbose: print("added", len(new_observations), "observations ( total", len(self.observations), ")")
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
        """
        Calculate weights for state pairs

        :param prev_states (Tensor): previous states
        :param new_states (Tensor): new states
        :param rewards (Tensor): rewards
        :return tensor containing weights of state pairs
        """
        return torch.exp((rewards - self.value_model(prev_states) + self.value_model(new_states))/self.value_model.eta)

    def split_policy_observations(self, ratio):
        prev_states = self.observations[:][:, 0].view(len(self.observations), 1)
        actions = self.observations[:][:, 1].view(len(self.observations), 1)
        rewards = self.observations[:][:, 2].view(len(self.observations), 1)
        new_states = self.observations[:][:, 3].view(len(self.observations), 1)
        weights = self.calc_weights(prev_states, new_states, rewards)

        observation_size = weights.size()[0]
        val_size = round(ratio*observation_size)
        train_size = observation_size - val_size

        train_dataset = torch.utils.data.TensorDataset(
            prev_states[val_size:],
            actions[val_size:],
            torch.FloatTensor(weights.data[val_size:]))

        valid_dataset = [
            prev_states[:val_size - 1],
            actions[:val_size - 1],
            weights[:val_size - 1]]

        return train_dataset, valid_dataset

    def improve_policy(self, learning_rate=1e-1, validation_ratio=0.1, batch_size=16):
        # load datasets
        train_dataset, valid_dataset = self.split_policy_observations(validation_ratio)
        data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer_mu = optim.Adagrad(self.policy_model.mu_net.parameters(), lr=learning_rate)
        optimizer_sigma = optim.Adagrad(self.policy_model.sigma_net.parameters(), lr=learning_rate)
        # train on batches
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epochs_opt_no_decrease < 5):
            for batch_idx, batch in enumerate(data_loader_train):
                optimizer_mu.zero_grad()
                optimizer_sigma.zero_grad()
                mu = self.policy_model.mu_net(batch[0])
                sigma = self.policy_model.get_sigma(batch[0])
                loss = NormalPolicyLoss_1D(mu, sigma, batch[1],
                                           batch[2])
                # backpropagate
                loss.backward()

                optimizer_mu.step()
                optimizer_sigma.step()
            # evaluate optimization iteration
            mu = self.policy_model.get_mu(valid_dataset[0])
            sigma = self.policy_model.get_sigma(valid_dataset[0])
            cur_loss_opt = NormalPolicyLoss_1D(mu, sigma, valid_dataset[1], valid_dataset[2])
            print(epoch_opt, "epoch, loss:", cur_loss_opt)
            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                epochs_opt_no_decrease = 0
            else:
                epochs_opt_no_decrease += 1
            last_loss_opt = cur_loss_opt
            epoch_opt += 1

    def improve_values(self, max_epochs_opt=50, episodes=10, timesteps=10, batch_size=100, learning_rate=1e-2, epsilon=0.1):
        # sanity check
        # init optimizer
        optimizer = optim.Adagrad(self.value_model.parameters(), lr=learning_rate)
        # explore with timesteps/episode
        self.explore(episodes, timesteps)
        # train on observation batches
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epoch_opt < max_epochs_opt) and (epochs_opt_no_decrease < 5):
            data_loader = DataLoader(self.observations, batch_size=batch_size, shuffle=True, num_workers=4)
            for batch_idx, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss = self.calc_loss(batch, batch_size, epsilon)
                # backpropagate
                loss.backward()
                optimizer.step()
            # evaluate optimization iteration
            cur_loss_opt = self.calc_loss(self.observations[:], len(self.observations), epsilon)
            if self.verbose: print("epoch:", epoch_opt+1, "of", max_epochs_opt, "with loss:", cur_loss_opt)

            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                epochs_opt_no_decrease = 0
            else:
                epochs_opt_no_decrease += 1
            last_loss_opt = cur_loss_opt
            epoch_opt += 1

def main():
    from environments.lqr import LQR
    from models.rand import Random
    from models.simple import Simple

    import random
    random.seed(42)

    environment = LQR(-2, 2)
    policy_model = PolicyNormal([1, 1])
    value_model = Simple()

    agent = Agent(environment, policy_model, value_model, verbose=True)
    agent.improve_values(episodes=1000, timesteps=50)
    #print([ p for p in agent.value_model.parameters()])
    agent.improve_policy()

if __name__ == '__main__':
    main()
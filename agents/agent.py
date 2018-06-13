import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data import SARSDataset
from utils.loss import REPSLoss

import torch
from torch import Tensor
import torch.optim as optim
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
        state = Tensor([state])
        action_tensor = self.policy_model.get_action(state)
        action_array = action_tensor[0].numpy()
        return action_array

    def average_reward(self):
        return torch.mean(self.observations[:][2])

    def explore(self, episodes, timesteps, remove_old, render):
        """
        Explore the environment for t timesteps.

        :param episodes (int): number of episodes to explore
        :param timesteps (int): number of timesteps per episode
        :param remove_old (bool): flag determining whether to keep the old observations
        :return newly collected observations
        """
        # initialize exploration
        new_observations = []
        episode_done = False
        cur_state = self.environment.reset()

        for t in range(episodes*timesteps):
            # reset environment
            if (t % timesteps == 0) or episode_done:
                cur_state = self.environment.reset()
            # perform action according to policy
            cur_action = self.get_action(cur_state)


            if render: self.environment.render()

            new_state, new_reward, episode_done, info = self.environment.step(cur_action)

            # save new observation
            new_observations.append({
                    'prev_state': cur_state,
                    'action': cur_action,
                    'reward': new_reward,
                    'new_state': new_state})
            # iterate:,
            cur_state = new_state

        # check for adding mode
        if remove_old:
            self.observations = SARSDataset(new_observations)
        else:
            self.observations.append(new_observations)

        if self.verbose: print("[explore] added", len(new_observations), "observations ( total", len(self.observations), ")")
        return new_observations

    def calc_loss(self, batch, batch_size, epsilon):
        # forward pass
        prev_states = batch[0]
        new_states = batch[3]
        prev_value_predictions = self.value_model(prev_states)
        new_value_predictions = self.value_model(new_states)

        # calculate loss
        rewards = batch[2]
        return REPSLoss(epsilon, self.value_model.eta, prev_value_predictions, new_value_predictions, rewards)

    def calc_weights(self, prev_states, new_states, rewards):
        """
        Calculate weights for state pairs

        :param prev_states (Tensor): previous states
        :param new_states (Tensor): new states
        :param rewards (Tensor): rewards
        :return tensor containing weights of state pairs
        """
        # TODO: use exp trick to avoid numerical instability
        norm_bellman_error = (rewards - self.value_model(prev_states) + self.value_model(new_states))/self.value_model.eta
        return torch.exp(norm_bellman_error - torch.max(norm_bellman_error))

    def improve_policy(self, learning_rate, val_ratio, batch_size):
        # load datasets
        prev_states = self.observations[:][0]
        actions = self.observations[:][1]
        rewards = self.observations[:][2]
        new_states = self.observations[:][3]
        weights = self.calc_weights(prev_states, new_states, rewards)

        # prepare training and validation splits
        observation_size = weights.size()[0]
        val_size = round(val_ratio*observation_size)
        train_size = observation_size - val_size
        train_dataset = torch.utils.data.TensorDataset(
            prev_states[val_size:],
            actions[val_size:],
            torch.Tensor(weights.data[val_size:]))
        val_dataset = [
            prev_states[:val_size - 1],
            actions[:val_size - 1],
            weights[:val_size - 1]]

        # optimize
        self.policy_model.optimize(train_dataset, val_dataset, batch_size, learning_rate, verbose=self.verbose)

    def improve_values(self, max_epochs_opt, batch_size, learning_rate, epsilon):
        # init optimizer
        optimizer = optim.Adagrad(self.value_model.parameters(), lr=learning_rate)
        # train on observation batches
        data_loader = DataLoader(self.observations, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epoch_opt < max_epochs_opt) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss = self.calc_loss(batch, batch_size, epsilon)
                # backpropagate
                loss.backward()
                optimizer.step()
            # evaluate optimization iteration
            cur_loss_opt = self.calc_loss(self.observations[:], len(self.observations), epsilon)
            if self.verbose:
                sys.stdout.write('\r[value] epoch: %d / %d | loss: %f' % (epoch_opt+1, max_epochs_opt, cur_loss_opt))
                sys.stdout.flush()

            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                best_model = self.value_model.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = cur_loss_opt
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1
        # use best previously found model
        self.value_model.load_state_dict(best_model)
        if self.verbose: sys.stdout.write('\r[value] training complete (%d epochs, %f best loss)' % (max_epochs_opt, last_loss_opt) + (' ' * (len(str(max_epochs_opt)))*2 + '\n'))


    def run_reps(self, iterations=10, exp_episodes=100, exp_timesteps=50, exp_remove_old=True, exp_render=False,
                 val_epochs=50, val_batch_size=100, val_lr=1e-2, val_epsilon=.1,
                 pol_lr=1e-2, pol_val_ratio=.1, pol_batch_size=100):
        for i in range(iterations):
            self.explore(episodes=exp_episodes, timesteps=exp_timesteps, remove_old=exp_remove_old, render=exp_render)
            self.improve_values(max_epochs_opt=val_epochs, batch_size=val_batch_size, learning_rate=val_lr, epsilon=val_epsilon)
            self.improve_policy(learning_rate=pol_lr, val_ratio=pol_val_ratio, batch_size=pol_batch_size)
            print(self.policy_model.sigma)
            if self.verbose: print("[reps] iteration", i+1, "/", iterations, "| average reward:", self.average_reward().data)

def main():
    from environments.lqr import LQR
    #from models.rand import Random
    from models.simple import Simple

    import random
    # random.seed(42)
    #
    # environment = LQR(-2, 2)
    # #policy_model = PolicyNormal([1, 1])
    # value_model = Simple()
    #
    # #agent = Agent(environment, policy_model, value_model, verbose=True)
    # agent.improve_values(episodes=1000, timesteps=5)
    # #print([ p for p in agent.value_model.parameters()])
    # agent.improve_policy()


if __name__ == '__main__':
    main()
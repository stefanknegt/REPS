from utils.data import *
from values.value_model import ValueModel
from policies.policy_model import PolicyModel

import gym
import random
import numpy as np

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

class Controller:
    """
    Controller for RL agents
    """

    def __init__(self, env_name, policy_model, value_model, action_min, action_max, verbose=False):
        """
        Constructor of agent
        """
        # init gym environment
        self.environment = gym.make(env_name)
        # init policy and value models
        self.policy_model = policy_model
        self.action_min = action_min
        self.action_max = action_max
        self.value_model = value_model

        # list of [SARSDataset, ...] observation batches
        self.observations = []

        # verbosity flag
        self.verbose = verbose


    def get_action(self, state):
        """
        Get action given current state according to the current policy.

        :param state: appropriate representation of the current state
        :return appriate action representation
        """
        state = Tensor([state])
        action_tensor = self.policy_model.get_action(state)[0]
        action_array = action_tensor[0].detach().numpy()
        action = self.scale_action(action_array)
        return action


    def scale_action(self, action):
        '''
        Takes an action [-1, 1] and returns an action [self.action_min, self.action_max]
        '''
        diff = self.action_max - self.action_min
        return self.action_min + (diff / 2) * (action + 1)


    def average_reward(self):
        res = 0.
        for obs in self.observations:
            res += torch.mean(obs[:][2])
        return res/len(self.observations)


    def explore(self, episodes, timesteps, trajectories, render):
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
            if (t % timesteps == 0) or episode_done or (random.uniform(0, 1) < trajectories/timesteps):
                cur_state = self.environment.reset()
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            new_state, new_reward, episode_done, info = self.environment.step(cur_action)
            if render: self.environment.render()

            # save new observation
            new_observations.append({
                    'prev_state': cur_state,
                    'action': cur_action,
                    'reward': new_reward,
                    'new_state': new_state})
            # iterate:,
            cur_state = new_state

        # add new observations
        self.observations.append(SARSDataset(new_observations))

        if self.verbose: print("[explore] added", len(new_observations), "observations | average reward:", float(self.average_reward()))
        return new_observations


    def get_observation_history(self, cur_iteration, history_size):
        res_observations = SARSDataset()
        for i in range(max(cur_iteration-history_size, 0), cur_iteration+1):
            res_observations.concatenate(self.observations[i])
        return res_observations


    def get_observation_split(self, observations, val_ratio):
        # load columns
        prev_states = observations[:][0]
        actions = observations[:][1]
        rewards = observations[:][2]
        new_states = observations[:][3]
        weights = self.value_model.get_weights(prev_states, new_states, rewards)
        check_values(weights=weights)

        # prepare training and validation splits
        observation_size = weights.size()[0]
        val_size = round(val_ratio*observation_size)
        train_size = observation_size - val_size
        train_dataset = torch.utils.data.TensorDataset(
            prev_states[val_size:],
            actions[val_size:],
            rewards[val_size:],
            new_states[val_size:],
            torch.Tensor(weights.data[val_size:]))
        val_dataset = [
            prev_states[:val_size - 1],
            actions[:val_size - 1],
            rewards[:val_size - 1],
            new_states[:val_size - 1],
            weights[:val_size - 1]]

        return train_dataset, val_dataset


    def optimize_model(self, model, train_obs, val_obs, max_epochs_opt, batch_size):
        # check model type
        mode = 'value' if isinstance(model, ValueModel) else 'policy'
        # train on observation batches
        data_loader = DataLoader(train_obs, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        while (epoch_opt < max_epochs_opt) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                prev_states = batch[:][0]
                actions = batch[:][1]
                rewards = batch[:][2]
                new_states = batch[:][3]
                weights = batch[:][4]
                # back prop steps
                if mode == 'value':
                    model.back_prop_step(prev_states, new_states, rewards)
                elif mode == 'policy':
                    model.back_prop_step(prev_states, actions, weights)
            # evaluate optimization iteration
            if mode == 'value':
                cur_loss_opt = model.get_loss(val_obs[0], val_obs[3], val_obs[2])
            elif mode == 'policy':
                cur_loss_opt = model.get_loss(val_obs[0], val_obs[1], val_obs[4])
            if self.verbose:
                sys.stdout.write('\r[%s] epoch: %d / %d | loss: %f' % (mode, epoch_opt+1, max_epochs_opt, cur_loss_opt))
                sys.stdout.flush()

            if (last_loss_opt is None) or (cur_loss_opt < last_loss_opt):
                best_model = model.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = cur_loss_opt
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1
        # use best previously found model
        model.load_state_dict(best_model)
        if self.verbose: sys.stdout.write('\r[%s] training complete (%d epochs, %f best loss)' % (mode, epoch_opt, last_loss_opt) + (' ' * (len(str(max_epochs_opt)))*2 + '\n'))


    def train(self, iterations=10, batch_size=100, eval_ratio=.1,
                exp_episodes=100, exp_timesteps=50, exp_trajectories=10, exp_history=5, exp_render=False,
                val_epochs=50, pol_epochs=100):
        for i in range(iterations):
            if self.verbose: print("[reps] iteration", i+1, "/", iterations)
            # explore and generate observation history
            self.explore(episodes=exp_episodes, timesteps=exp_timesteps, trajectories=exp_trajectories, render=exp_render)
            observation_history = self.get_observation_history(i, exp_history)
            train_observations, val_observations = self.get_observation_split(observation_history, eval_ratio)
            # run value model improvement
            self.optimize_model(self.value_model, train_obs=train_observations, val_obs=val_observations, max_epochs_opt=val_epochs, batch_size=batch_size)
            # run policy model improvement
            self.optimize_model(self.policy_model, train_obs=train_observations, val_obs=val_observations, max_epochs_opt=pol_epochs, batch_size=batch_size)

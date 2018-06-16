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
        action_tensor = self.policy_model.get_action(state)
        action_array = action_tensor[0].detach().numpy()
        return action_array


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


    def last_average_reward(self):
        return torch.mean(self.observations[-1][:][2])


    def set_cumulative_sum(self, obs_dicts, episode_length, gamma):
        episode_start_idx = len(obs_dicts) - episode_length
        episode_end_idx = len(obs_dicts) - 1
        cum_sum = 0.
        for disc_pow, obs_idx in enumerate(range(episode_end_idx, episode_start_idx-1, -1)):
            cum_sum += obs_dicts[obs_idx]['reward'] * (gamma ** disc_pow)
            obs_dicts[obs_idx]['cum_sum'] = cum_sum


    def explore(self, episodes, timesteps, reset_prob, gamma, render):
        # initialize exploration
        new_observations = []
        episode_t = 0
        episode_done = False
        cur_state = self.environment.reset()

        for t in range(episodes*timesteps):
            # reset environment
            if (t % timesteps == 0) or episode_done or (random.uniform(0, 1) < reset_prob):
                # calculate cumulative sum
                self.set_cumulative_sum(new_observations, episode_t, gamma)
                episode_t = 0
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
                    'new_state': new_state,
                    'cum_sum': 0.})
            # iterate
            episode_t += 1
            cur_state = new_state

        # add new observations
        self.observations.append(SARSDataset(new_observations))

        if self.verbose: print("[explore] added %d observations | average reward: %f" % (len(new_observations), float(self.last_average_reward())))
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
        cum_sums = observations[:][4]
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
            cum_sums[val_size:],
            torch.Tensor(weights.data[val_size:]))
        val_dataset = [
            prev_states[:val_size - 1],
            actions[:val_size - 1],
            rewards[:val_size - 1],
            new_states[:val_size - 1],
            cum_sums[:val_size - 1],
            weights[:val_size - 1]]

        return train_dataset, val_dataset


    def optimize_model(self, model, train_obs, val_obs, max_epochs_opt, batch_size, is_init=False):
        # check model type
        mode = 'value' if isinstance(model, ValueModel) else 'policy'
        mode = 'value_init' if is_init else mode
        # train on observation batches
        data_loader = DataLoader(train_obs, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0
        # monte carlo initialisation
        while (epoch_opt < max_epochs_opt) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                prev_states = batch[:][0]
                actions = batch[:][1]
                rewards = batch[:][2]
                new_states = batch[:][3]
                cum_sums = batch[:][4]
                weights = batch[:][5]
                # back prop steps
                if mode == 'value':
                    model.back_prop_step(prev_states, new_states, rewards)
                elif mode == 'policy':
                    model.back_prop_step(prev_states, actions, weights)
                elif mode == 'value_init':
                    model.init_back_prop_step(prev_states, cum_sums)

            # evaluate optimization iteration
            if mode == 'value':
                cur_loss_opt = model.get_loss(val_obs[0], val_obs[3], val_obs[2])
            elif mode == 'policy':
                cur_loss_opt = model.get_loss(val_obs[0], val_obs[1], val_obs[5])
            if mode == 'value_init':
                cur_loss_opt = model.get_init_loss(val_obs[0], val_obs[4])
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
                exp_episodes=100, exp_timesteps=50, exp_reset_prob=.0, exp_history=5, exp_gamma=.9, exp_render=False,
                val_epochs=50, pol_epochs=100):
        for i in range(iterations):
            if self.verbose: print("[reps] iteration", i+1, "/", iterations)
            # explore and generate observation history
            self.explore(episodes=exp_episodes, timesteps=exp_timesteps, reset_prob=exp_reset_prob, gamma=exp_gamma, render=exp_render)
            observation_history = self.get_observation_history(i, exp_history)
            # run initial value model optimisation using cumulative sums
            train_observations, val_observations = self.get_observation_split(observation_history, eval_ratio)
            if i == 0:
                self.optimize_model(self.value_model, train_obs=train_observations, val_obs=val_observations, max_epochs_opt=val_epochs, batch_size=batch_size, is_init=True)
            # run value model improvement
            self.optimize_model(self.value_model, train_obs=train_observations, val_obs=val_observations, max_epochs_opt=val_epochs, batch_size=batch_size, is_init=False)
            # run policy model improvement using updated weights
            train_observations, val_observations = self.get_observation_split(observation_history, eval_ratio)
            self.optimize_model(self.policy_model, train_obs=train_observations, val_obs=val_observations, max_epochs_opt=pol_epochs, batch_size=batch_size, is_init=False)

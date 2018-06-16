from utils.data import *

import gym
import random
import numpy as np

import torch
from torch import Tensor
from utils.average_env import make_average_env
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

class Controller:
    """
    Controller for RL agents
    """

    def __init__(self, env_name, policy_model, value_model, reset_prob, history_depth=1, verbose=False):
        """
        Constructor of agent
        """
        # init gym environment
        self.eval_env = gym.make(env_name)
        self.sample_env = make_average_env(self.eval_env, reset_prob)

        # get max and min actions
        self.max_action = np.asscalar(self.eval_env.action_space.high[0])
        self.min_action = -np.asscalar(self.eval_env.action_space.low[0])
        assert self.max_action == -self.min_action

        # init policy and value models
        self.policy_model = policy_model
        self.value_model = value_model

        # list of [SARSDataset, ...] observation batches
        self.observations = []

        # verbosity flag
        self.verbose = verbose

        # gamma for values of states
        self.gamma = 1 - reset_prob

        # history depth
        self.history_depth = history_depth

        # initial states
        self.init_states = []

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

    def set_cumulative_sum(self, obs_dicts, episode_length, gamma):
        episode_start_idx = len(obs_dicts) - episode_length
        episode_end_idx = len(obs_dicts) - 1
        cum_sum = 0.
        for disc_pow, obs_idx in enumerate(range(episode_end_idx, episode_start_idx-1, -1)):
            cum_sum += obs_dicts[obs_idx]['reward'] * (gamma ** disc_pow)
            obs_dicts[obs_idx]['cum_sum'] = cum_sum

    def explore(self, episodes, max_timesteps_per_episode, gamma_discount):
        # initialize exploration
        new_observations = []
        episode_t = 0
        episode_done = False
        cur_state = self.sample_env.reset()

        for t in range(episodes*max_timesteps_per_episode):
            # reset environment
            if (t % max_timesteps_per_episode == 0) or episode_done:
                # calculate cumulative sum
                self.set_cumulative_sum(new_observations, episode_t, gamma_discount)
                episode_t = 0
                cur_state = self.sample_env.reset()
                # add starting state to init states
                self.init_states.append(cur_state)
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            new_state, new_reward, episode_done, info = self.sample_env.step(cur_action)

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

        # remove old and add new observations
        if len(self.observations) == self.history_depth:
            self.observations = self.observations[1:]
        self.observations.append(SARSDataset(new_observations))

        if self.verbose:
            print("[explore] added %d observations " % (len(new_observations)))
        return new_observations

    def get_observation_history(self):
        res_observations = SARSDataset()
        for i in range(len(self.observations)):
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
            weights[val_size:])
        val_dataset = [
            prev_states[:val_size - 1],
            actions[:val_size - 1],
            rewards[:val_size - 1],
            new_states[:val_size - 1],
            cum_sums[:val_size - 1],
            weights[:val_size - 1]]

        return train_dataset, val_dataset

    # TODO Everything below - write main loop and optimizers in here
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

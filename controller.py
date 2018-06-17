from utils.data import *

import gym
import random
import numpy as np

import torch
import torch.random
from torch import Tensor
from utils.average_env import make_average_env
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from values.mlp import MLPValue
from policies.normal_mlp import MLPNormalPolicy

class Controller:
    """
    Controller for RL agents
    """

    def __init__(self, env_name, policy_model : MLPNormalPolicy, value_model : MLPValue, reset_prob, history_depth=1, verbose=False, random_seed=42):
        """
        Constructor of agent
        """
        # init gym environment
        env = gym.make(env_name)
        self.eval_env = env
        self.sample_env = make_average_env(env, reset_prob)

        # get max and min actions
        self.max_action = np.asscalar(self.eval_env.action_space.high[0])
        self.min_action = np.asscalar(self.eval_env.action_space.low[0])
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

        # set random seeds
        if random_seed >= 0:
            self.set_seeds(random_seed)

    def set_seeds(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)
        self.sample_env.seed(seed)
        self.eval_env.seed(seed)

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
        weights = self.value_model.get_weights(prev_states, new_states, Tensor(self.init_states), rewards, self.gamma)

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

    def evaluate(self, episodes=10, max_timesteps_per_episode=100, render=True):
        cur_state = self.eval_env.reset()
        avg_rewards = []
        rewards = []
        episode_done = False
        for t in range(episodes*max_timesteps_per_episode):
            if (t % max_timesteps_per_episode == 0) or episode_done:
                # reset environment
                cur_state = self.sample_env.reset()
                avg_rewards.append(np.mean(np.array(rewards)))
                rewards = []
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            cur_state, new_reward, episode_done, info = self.sample_env.step(cur_action)
            rewards.append(new_reward)
            if render:
                self.eval_env.render()
        return np.mean(np.array(avg_rewards))

    def train(self, iterations=10, batch_size=64, val_ratio=.1,
                exp_episodes=10, exp_timesteps=100, exp_history=1, exp_gamma_discount=1,
                val_epochs=50, pol_epochs=100):
        # Reset mu weights (to make mean around 0 at start)
        self.policy_model.reset()
        avg_reward_over_time = []

        for i in range(iterations):
            # Gather data
            if i == 0:
                for j in range(exp_history):
                    self.explore(exp_episodes, exp_timesteps, exp_gamma_discount)
            else:
                self.explore(exp_episodes, exp_timesteps, exp_gamma_discount)

            if i == 0:
                train_dataset, val_dataset = self.get_observation_split(self.get_observation_history(), val_ratio)
                # Optimize V and eta
                self.value_model.optimize_loss(train_dataset, val_dataset, loss_type=self.value_model.mse_loss,
                                               optimizer=self.value_model.optimizer_all, max_epochs=val_epochs,
                                              batch_size=batch_size, gamma=0, verbose=True)


                print(self.value_model.eta)
                # Optimize only eta (no batches)
                self.value_model.optimize_loss(train_dataset, val_dataset, loss_type=self.value_model.reps_loss,
                                               optimizer=self.value_model.optimizer_eta, max_epochs=val_epochs,
                                              batch_size=-1, gamma=self.gamma, verbose=True, init_states=Tensor(self.init_states))
                print(self.value_model.eta)

            train_dataset, val_dataset = self.get_observation_split(self.get_observation_history(), val_ratio)

            obs = self.get_observation_history()
            prev_states = obs[:][0]
            actions = obs[:][1]
            rewards = obs[:][2]
            new_states = obs[:][3]

            for itr in range(20):
                self.value_model.optimize_loss(train_dataset, val_dataset, loss_type=self.value_model.reps_loss,
                                               optimizer=self.value_model.optimizer_all, max_epochs=val_epochs,
                                               batch_size=batch_size, gamma=self.gamma, verbose=True, init_states=Tensor(self.init_states))

                w = self.value_model.get_weights(prev_states,new_states, Tensor(self.init_states),rewards, self.gamma, True)
                w = w.detach().numpy()
                kl = np.nansum(w[np.nonzero(w)] * np.log(w[np.nonzero(w)] * w.size))
                kl_err = np.abs(kl - self.value_model.epsilon)
                kl_tol= 0.1
                valid_kl = kl_err < kl_tol * self.value_model.epsilon
                print("KL: ", kl, " eta: ", self.value_model.eta)

                if valid_kl and (itr > 10):
                    break

                if itr > 10 and not kl < self.value_model.epsilon :
                    self.value_model.optimize_loss(train_dataset, val_dataset, loss_type=self.value_model.reps_loss,
                                               optimizer=self.value_model.optimizer_eta, max_epochs=val_epochs,
                                              batch_size=-1, gamma=self.gamma, verbose=True, init_states=Tensor(self.init_states))

            train_dataset, val_dataset = self.get_observation_split(self.get_observation_history(), val_ratio)
            self.policy_model.optimize_loss(train_dataset, val_dataset, pol_epochs, batch_size, True)

            self.evaluate()





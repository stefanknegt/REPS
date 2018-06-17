from utils.data import *
from utils.average_env import make_average_env

import gym
import random
import numpy as np

import torch
import torch.random
import torch.optim as optim
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

class Controller:
    """
    Controller for RL agents
    """

    def __init__(self, env_name, policy_model, value_model, 
                reset_prob=0.02, history_depth=1, verbose=False):
        # init gym environment
        env = gym.make(env_name)
        self.env_eval = env
        self.env_sample = make_average_env(env, reset_prob)

        # get max and min actions
        self.max_action = np.asscalar(self.env_eval.action_space.high[0])
        self.min_action = np.asscalar(self.env_eval.action_space.low[0])
        assert self.max_action == -self.min_action

        # init policy and value models
        self.policy_model = policy_model
        self.value_model = value_model

        # list of [SARSDataset, ...] observation batches
        self.observations = []
        # initial states
        self.init_states = []
        # history depth
        self.history_depth = history_depth

        # set transition probability to initial state (gamma)
        self.gamma = 1 - reset_prob

        # verbosity flag
        self.verbose = verbose


    def set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.env_eval.seed(seed)
        self.env_sample.seed(seed)


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


    def explore(self, episodes, max_timesteps, gamma_discount):
        # initialize exploration
        new_observations = []
        episode_t = 0
        episode_done = False
        cur_state = self.env_sample.reset()

        for t in range(episodes*max_timesteps):
            # reset environment
            if (t % max_timesteps == 0) or episode_done:
                # calculate cumulative sum
                self.set_cumulative_sum(new_observations, episode_t, gamma_discount)
                episode_t = 0
                cur_state = self.env_sample.reset()
                # add starting state to init states
                self.init_states.append(cur_state)
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            new_state, new_reward, episode_done, info = self.env_sample.step(cur_action)

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


    def evaluate(self, episodes, max_timesteps, render):
        cur_state = self.env_eval.reset()
        avg_rewards = []
        rewards = []
        episode_done = False
        for t in range(episodes*max_timesteps):
            if (t % max_timesteps == 0) or episode_done:
                # reset environment
                cur_state = self.env_sample.reset()
                avg_rewards.append(np.mean(np.array(rewards)))
                rewards = []
            # perform action according to policy
            cur_action = self.get_action(cur_state)
            cur_state, new_reward, episode_done, info = self.env_sample.step(cur_action)
            rewards.append(new_reward)
            if render:
                self.env_eval.render()
        return np.mean(np.array(avg_rewards))


    def optimize(self, mode, model, optimizer, train_dataset, val_dataset, max_epochs, batch_size, verbose):
        if batch_size <= 0:
            batch_size = len(train_dataset)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        best_model = None
        last_loss_opt = None
        epochs_opt_no_decrease = 0
        epoch_opt = 0

        while (epoch_opt < max_epochs) and (epochs_opt_no_decrease < 3):
            for batch_idx, batch in enumerate(data_loader):
                prev_states = batch[:][0]
                actions = batch[:][1]
                rewards = batch[:][2]
                new_states = batch[:][3]
                cum_sums = batch[:][4]
                weights = batch[:][5]
                init_states = Tensor(self.init_states)

                # calculate loss
                if mode == 'value_init':
                    loss = model.mse_loss(prev_states, cum_sums)
                elif mode in ['value', 'eta']:
                    loss = model.reps_loss(prev_states, new_states, init_states, rewards, self.gamma)
                elif mode == 'policy':
                    loss = model.get_loss(prev_states, actions, weights)

                # propagate back
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calculate validation loss
            if mode == 'value_init':
                valid_loss = model.mse_loss(val_dataset[0], val_dataset[4])
            elif mode in ['value', 'eta']:
                valid_loss = model.reps_loss(val_dataset[0], val_dataset[3], init_states, val_dataset[2], self.gamma)
            elif mode == 'policy':
                valid_loss = model.get_loss(val_dataset[0], val_dataset[1], val_dataset[5])

            if verbose:
                sys.stdout.write('\r[%s] epoch: %d / %d | loss: %f' % (mode, epoch_opt+1, max_epochs, valid_loss))
                sys.stdout.flush()

            # check if loss is decreasing
            if (last_loss_opt is None) or (valid_loss < last_loss_opt):
                best_model = model.state_dict()
                epochs_opt_no_decrease = 0
                last_loss_opt = valid_loss
            else:
                epochs_opt_no_decrease += 1
            epoch_opt += 1

        # use best previously found model
        model.load_state_dict(best_model)
        if verbose:
            sys.stdout.write('\r[%s] training complete (%d epochs, %f best loss)' % (mode, epoch_opt, last_loss_opt) + (' ' * (len(str(max_epochs))) * 2 + '\n'))


    def check_KL(self):
        # prepare data
        obs = self.get_observation_history()
        prev_states = obs[:][0]
        actions = obs[:][1]
        rewards = obs[:][2]
        new_states = obs[:][3]
        init_states = Tensor(self.init_states)
        # calculate weights
        weights = self.value_model.get_weights(prev_states, new_states, init_states,rewards, self.gamma, normalized=True)
        weights = weights.detach().numpy()

        kl = np.nansum(weights[np.nonzero(weights)] * np.log(weights[np.nonzero(weights)] * weights.size))
        kl_err = np.abs(kl - self.value_model.epsilon)
        kl_tol= 0.1
        valid_kl = kl_err < kl_tol * self.value_model.epsilon

        return kl, valid_kl


    def train(self, iterations=10, batch_size=64, val_ratio=.1,
                exp_episodes=10, exp_timesteps=100, exp_gamma_discount=1,
                val_iterations=20, val_min_iterations=10, val_epochs=50, pol_epochs=100,
                eval_episodes=10, eval_timesteps=100, eval_render=True):
        # Reset mu weights (to make mean around 0 at start)
        self.policy_model.reset()

        for reps_i in range(iterations):
            if self.verbose: print("[REPS] iteration", reps_i+1, "/", iterations)

            # Gather and prepare data (minimum of history_depth explorations)
            for _ in range(max(1, self.history_depth - len(self.observations))):
                self.explore(exp_episodes, exp_timesteps, exp_gamma_discount)
            train_dataset, val_dataset = self.get_observation_split(self.get_observation_history(), val_ratio)

            # Special actions for initialisation iteration
            if reps_i == 0:
                # Optimize V and eta
                self.optimize(mode='value_init', model=self.value_model, optimizer=self.value_model.optimizer_all,
                                train_dataset=train_dataset, val_dataset=val_dataset,
                                max_epochs=val_epochs, batch_size=batch_size, verbose=self.verbose)
                # Optimize only eta (no batches)
                self.optimize(mode='eta', model=self.value_model, optimizer=self.value_model.optimizer_eta,
                                train_dataset=train_dataset, val_dataset=val_dataset,
                                max_epochs=val_epochs, batch_size=-1, verbose=self.verbose)
                print("[eta_init] initial eta: ", float(self.value_model.eta))

            # Value optimization
            for val_i in range(val_iterations):
                print("[value] iteration", val_i+1, "/", val_iterations)
                self.optimize(mode='value', model=self.value_model, optimizer=self.value_model.optimizer_all,
                                train_dataset=train_dataset, val_dataset=val_dataset,
                                max_epochs=val_epochs, batch_size=batch_size, verbose=self.verbose)

                kl, valid_kl = self.check_KL()
                print("[value] KL:", kl, "| eta:", float(self.value_model.eta))

                if valid_kl and (val_i > val_min_iterations):
                    break

                if (val_i > val_min_iterations) and (kl >= self.value_model.epsilon) :
                    self.optimize(mode='eta', model=self.value_model, optimizer=self.value_model.optimizer_eta,
                                    train_dataset=train_dataset, val_dataset=val_dataset,
                                    max_epochs=val_epochs, batch_size=-1, verbose=self.verbose)

            # Policy optimization
            # recalculate weights
            train_dataset, val_dataset = self.get_observation_split(self.get_observation_history(), val_ratio)
            self.optimize(mode='policy', model=self.policy_model, optimizer=self.policy_model.optimizer,
                            train_dataset=train_dataset, val_dataset=val_dataset,
                            max_epochs=pol_epochs, batch_size=batch_size, verbose=self.verbose)

            # Evaluation
            avg_reward = self.evaluate(episodes=eval_episodes, max_timesteps=eval_timesteps, render=eval_render)
            print("[eval] average reward:", avg_reward)
            print()





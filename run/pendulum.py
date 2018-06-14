import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.policy_model import Policy
from values.value_model import Value
from utils.check_env import environment_check

name = 'Pendulum-v0'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = Policy(state_dim, action_dim, hidden_dim, mu_range=0.4, sigma_range=1, lr=0.01)
value_model = Value(state_dim, hidden_dim, value_range=1, eta=0.1, epsilon=0.5, lr=0.01)

model = Controller(name, policy_model, value_model, action_min, action_max, verbose=True)
model.train(iterations=10, batch_size=100,
            exp_episodes=10, exp_timesteps=100, exp_trajectories=10, exp_history=5, exp_render=False)

policy_model.save('run/' + name + '.pth')

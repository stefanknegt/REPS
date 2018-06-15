import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal import NormalPolicy
from policies.normal_mlp import NormalPolicy_MLP
from values.linear import LinearValue
from utils.check_env import environment_check

name = 'Pendulum-v0'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = NormalPolicy_MLP([3, 4, 5, 1], 1e-3, 1, act_bound=2)
value_model = LinearValue(state_dim, hidden_dim, value_range=1, eta=0.1, epsilon=0.5, lr=0.01)

model = Controller(name, policy_model, value_model, action_min, action_max, verbose=True)
model.train(iterations=10, batch_size=100,
            exp_episodes=10, exp_timesteps=100, exp_trajectories=10, exp_history=5, exp_render=False)

policy_model.save('run/' + name + '.pth')

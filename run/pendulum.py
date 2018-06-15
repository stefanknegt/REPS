import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import NormalPolicy_MLP
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Reacher-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = NormalPolicy_MLP([11, 4, 5, 2], 1e-3, 1, act_bound=2)
value_model = MLPValue([state_dim, hidden_dim, 1])

model = Controller(name, policy_model, value_model, action_min, action_max, verbose=True)
model.train(iterations=10, batch_size=100,
            exp_episodes=10, exp_timesteps=100, exp_trajectories=10, exp_history=5, exp_render=False)

policy_model.save('run/' + name + '.pth')
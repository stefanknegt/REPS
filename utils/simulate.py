import torch
from torch.autograd import Variable
import gym
import mujoco_py

policy_model = torch.load('test.pth')

env = gym.make('HalfCheetah-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        _,_,action = the_model.get_action(Variable(torch.transpose(torch.tensor([observation]).float(),1,0)))
        new_action = action.detach().numpy().flatten()
        observation, reward, done, info = env.step(new_action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

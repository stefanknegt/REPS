import gym

def environment_check(name):
    '''
    This function checks the gym environment to get the action and state space.
    Also it returns the min and max value of the actions.
    '''
    env = gym.make(name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])

    return state_dim, action_dim, action_min, action_max

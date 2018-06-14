import gym

class AtariBreakout():
    """
    Wrapper class for Arcade Learning Environment

    Available actions:
        0: NOP, 1: BTN, 2: R, 3: L
    States:
        list of size 128 with bytes[[0, 255], ... ]
        episode ends with game over (ball out of bounds or no lives left)
    """
    def __init__(self):
        self.ale = gym.make('Breakout-ram-v0')

    def step(self, action):
        return self.ale.step(int(action.squeeze()))

    def reset(self):
        return self.ale.reset()

    def render(self):
        return self.ale.render()
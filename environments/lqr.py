import random

class LQR:
    '''
    This is the class for the Linear Quadratic Rewards. Upon initialization it initializes
    random values for the state transistions and rewards. Then the functions can be called using the Step and Reset functions.
    '''
    def __init__(self, range_min, range_max):
        #Initialize parameters for transistions and rewards:
        self.range_min = range_min
        self.range_max = range_max
        self.C1 = 1#random.random()
        self.C2 = 1#random.random()
        self.C3 = 1#random.random()
        self.C4 = 1   #random.random()

        #Initialize a random starting state
        self.reset()

    def step(self, action):
        #We calculate the new state using the parameters:
        self.state = self.C3 * self.state + self.C4 * action
        # self.state = max([self.state, self.range_min])
        # self.state = min([self.state, self.range_max])

        self.reward = - self.C1 * (self.state ** 2) - self.C2 * (action ** 2)
        return self.state, self.reward, False, {}

    def reset(self):
        self.state = random.uniform(self.range_min, self.range_max)
        return self.state


def main():
    #Initialize an environment
    env = LQR(-10, 10) #you can select the range of states

    #Get a batch of data for random samples
    batch_size = 10
    batch_data = []
    for s in range(batch_size):
        #We save all data in dicts
        data = {}
        data['prev_state'] = env.state
        data['action'] = random.uniform(-1,1)
        data['new_state'], data['reward'] = env.step(data['action']) 

        #Put the dict in a list:
        batch_data.append(data)

    print(batch_data)
    return batch_data

if __name__ == "__main__":
    main()
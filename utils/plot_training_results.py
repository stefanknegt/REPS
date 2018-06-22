import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_pickled_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def plot_results(x,y):
    line = plt.plot(x,y)
    #plt.setp(line, linestyle='--')       # set both to dashed
    plt.setp(line, linewidth=3, color='b')
    plt.title("Average reward during training for Pendulum task",fontsize=16)
    plt.xlabel("Iteration",fontsize=14)
    plt.ylabel("Reward",fontsize=14)
    plt.show()

data = load_pickled_data('../run/results/Pendulum-v0_v0.pickle')
x = range(1,len(data['rewards'])+1)
y = data['rewards']
plot_results(x,y)

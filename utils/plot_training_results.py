import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_pickled_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def plot_results(x1,y,title,xlabel,ylabel,names,y2=None,y3=None,y4=None,label_fontsize=14):
    if y2 == None:
        line = plt.plot(x,y)
        #plt.setp(line, linestyle='--')       # set both to dashed
        plt.setp(line, linewidth=3, color='b')
        plt.title(title,fontsize=18)
        plt.xlabel(xlabel,fontsize=label_fontsize)
        plt.ylabel(ylabel,fontsize=label_fontsize)
        plt.show()
    elif y != None and y2 != None and y3 == None:
        lines = plt.plot(x,y,x,y2)
        l1,l2 = lines
        #plt.setp(lines, linestyle='--')       # set both to dashed
        plt.setp(l1, linewidth=3, color='b')
        plt.setp(l2, linewidth=3, color='r')
        plt.title("Average reward during training for Pendulum task",fontsize=18)
        plt.xlabel("Iteration",fontsize=label_fontsize)
        plt.ylabel("Reward",fontsize=label_fontsize)
        plt.legend(names)
        plt.show()
    elif y4 == None:
        lines = plt.plot(x,y,x,y2,x,y3)
        l1,l2,l3 = lines
        #plt.setp(lines, linestyle='--')       # set both to dashed
        plt.setp(l1, linewidth=3, color='b')
        plt.setp(l2, linewidth=3, color='r')
        plt.setp(l3, linewidth=3, color='y')
        plt.title("Average reward during training for Pendulum task",fontsize=18)
        plt.xlabel("Iteration",fontsize=label_fontsize)
        plt.ylabel("Reward",fontsize=label_fontsize)
        plt.legend(names)
        plt.show()
    else:
        lines = plt.plot(x,y,x,y2,x,y3,x,y4)
        l1,l2,l3,l4 = lines
        #plt.setp(lines, linestyle='--')       # set both to dashed
        plt.setp(l1, linewidth=3, color='b')
        plt.setp(l2, linewidth=3, color='r')
        plt.setp(l3, linewidth=3, color='y')
        plt.setp(l4, linewidth=3, color='g')
        plt.title("Average reward during training for Pendulum task",fontsize=18)
        plt.xlabel("Iteration",fontsize=label_fontsize)
        plt.ylabel("Reward",fontsize=label_fontsize)
        plt.legend(names)
        plt.show()

data = load_pickled_data('../run/results/Pendulum-v0_v0.pickle')
x = range(1,len(data['rewards'])+1)
y = data['rewards']
y2 = data['eta']
y3 = data['value_loss']
y4 = data['policy_loss']
plot_results(x,y,title='Average reward during training for Pendulum task',xlabel='Iteration',ylabel='Reward',names=['Reward','Eta','Value loss', 'Policy loss'],y2=y2,y3=y3,y4=y4)

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

def plot_fill_between(x,y,xlabel,ylabel,names):
    y_final = []
    y_max = []
    y_min = []
    y_std = []
    for i in range(0,len(y[1])):
        y_final.append(np.mean(y[:,i]))
        y_max.append(np.max(y[:,i]))
        y_min.append(np.min(y[:,i]))
        y_std.append(np.std(y[:,i]))
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel(ylabel,fontsize=14)
    plt.xticks(np.arange(min(x), max(x)+1, 200000))
    plt.xlim(0,max(x))

    plt.plot(x, y_final, color='#1B2ACC',alpha=0.5,linewidth=2)

    plt.fill_between(x, np.subtract(y_final,y_std), np.add(y_final,y_std), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.savefig('Test2.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
"""
data1 = load_pickled_data('../results/paperResults/HalfCheetah-v2_v0.1_results.pickle')
data2 = load_pickled_data('../results/paperResults/HalfCheetah-v2_v0.2_results.pickle')
data3 = load_pickled_data('../results/paperResults/HalfCheetah-v2_v0.3_results.pickle')
data4 = load_pickled_data('../results/paperResults/HalfCheetah-v2_v0.4_results.pickle')
data5 = load_pickled_data('../results/paperResults/HalfCheetah-v2_v0.5_results.pickle')
"""

data1 = load_pickled_data('../results/paperResults/Swimmer-v2_eval_run_1_results.pickle')
data2 = load_pickled_data('../results/paperResults/Swimmer-v2_eval_run_2_results.pickle')
data3 = load_pickled_data('../results/paperResults/Swimmer-v2_eval_run_3_results.pickle')
data4 = load_pickled_data('../results/paperResults/Swimmer-v2_eval_run_4_results.pickle')
data5 = load_pickled_data('../results/paperResults/Swimmer-v2_eval_run_5_results.pickle')

Pendulum = False

timesteps = data1['timesteps_iteration']

if Pendulum == True:
    x = range(0,(len(data1['rewards'])+1)*timesteps,timesteps)
    y_init = [-1256.0]
    y1 = y_init + (data1['rewards'])
    max_y1 = max(y1)
    print(max_y1)
    y2 = y_init + (data2['rewards'])
    max_y2 = max(y2)
    print(max_y2)
    y3 = y_init + (data3['rewards'])
    max_y3 = max(y3)
    print(max_y3)
    y4 = y_init + (data4['rewards'])
    max_y4 = max(y4)
    print(max_y4)
    y5 = y_init + (data5['rewards'])
    max_y5 = max(y5)
    print(max_y5)
else:
    x = range(0,(len(data1['rewards']))*timesteps,timesteps)
    y1 = data1['rewards']
    y2 = data2['rewards']
    y3 = data3['rewards']
    y4 = data4['rewards']
    y5 = data5['rewards']
    max_y1 = np.max(y1)
    max_y2 = np.max(y2)
    max_y3 = np.max(y3)
    max_y4 = np.max(y4)
    max_y5 = np.max(y5)


y = np.stack((y1,y2,y3,y4,y5))
print(y)
print((max_y1+max_y2+max_y3+max_y4+max_y5)/5)
print((max_y1,max_y2,max_y3,max_y4,max_y5))

plot_fill_between(x,y,'Timesteps','Reward',[''])

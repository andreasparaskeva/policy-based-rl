import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None, xlabel='Episode', ylabel='Reward'):
        self.fig,self.ax = plt.subplots(figsize=(8,5))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)      
        self.palette = itertools.cycle(sns.color_palette())
        self.offset = 0
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, y_std,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        size = len(y)
        error_interval = int(size / 10)
        self.offset += int(size / 10 / 5)
        if not type(y_std) == list:
            y_std = y_std / 2
            if label is not None:
                c = next(self.palette)
                # with sns.color_palette("Spectral", n_colors=10) as c:
                self.ax.plot(y, label=label, color=c)
                self.ax.errorbar(
                    range(self.offset, size, error_interval), 
                    y[self.offset:][::error_interval], 
                    yerr=y_std[self.offset:][::error_interval], 
                    fmt='.', 
                    color=c,
                    markersize=1, 
                    capsize=2,
                    elinewidth=0.5
                )
            else:
                self.ax.errorbar(range(len(y)), y, yerr=y_std, fmt='.', markersize=5, capsize=5, elinewidth=1)
        else:
            if label is not None:
                c = next(self.palette)
                # with sns.color_palette("Spectral", n_colors=10) as c:
                self.ax.plot(y, label=label, color=c)
                self.ax.errorbar(
                    range(self.offset, size, error_interval), 
                    y[self.offset:][::error_interval], 
                    yerr=[y_std[0][self.offset:][::error_interval], y_std[1][self.offset:][::error_interval]], 
                    fmt='.', 
                    color=c,
                    markersize=1, 
                    capsize=2,
                    elinewidth=0.5
                )
            # with sns.color_palette("Spectral", n_colors=10):
            # self.ax.fill_between(range(len(y)), y - y_std, y + y_std, alpha=0.2, label=label)
            
            # self.ax.plot(y,label=label)
       
            # self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png', loc='lower right'):
        ''' name: string for filename of saved figure '''
        self.ax.legend(loc=loc, ncol=1)
        # self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = np.array(x) / temp # scale by temperature
    z = x - np.max(x) # substract max to prevent overflow of softmax
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

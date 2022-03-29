import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
def plotLoss(loss:list = [], label:str = "Loss", ax = None,show:bool = True):
    x = np.arange(0,len(loss))
    assert len(loss)!=0
    if ax == None:
        _, ax = plt.subplots()
    ax.plot(x, loss, label = label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    if show: 
        plt.show()
    return ax
    
def plotAcc(TrainAcc:list = [],TestAcc:list = [], ax = None,show:bool = True):
    # x = np.arange(0,len(TrainAcc))
    assert len(TrainAcc) == len(TestAcc) !=0
    if ax == None:
        _, ax = plt.subplots()
    ax.plot(TestAcc, color = 'r', label = 'Test accuracy')
    ax.plot(TrainAcc, color = 'g', label = 'Train accuracy')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    if show: 
        plt.show()
    return ax

    
    
    
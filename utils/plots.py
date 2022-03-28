import matplotlib.pyplot as plt
import numpy as np
def plotLoss(loss:list = [], label:str = "Loss", ax = None,show:bool = True):
    x = np.arange(0,len(loss))
    assert len(loss)!=0
    if ax == None:
        _, ax = plt.subplots()
    ax.plot(x, loss, label = label)
    ax.set_xlabel("epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    if show: 
        plt.show
    return ax
    
def plotAcc(TrainAcc:list = [],TestAcc:list = [], ax = None,show:bool = True):
    x = np.arange(0,len(TrainAcc))
    assert len(TrainAcc) != len(TestAcc)
    if ax == None:
        _, ax = plt.subplots()
    ax.plot(x, TestAcc, color = 'r', label = 'Test accuracy')
    ax.plot(x, TrainAcc, color = 'g', label = 'Train accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    if show: 
        plt.show
    return ax

    
    
    
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from torchsummary import summary
def plotLoss(loss:list = [], label:str = "Loss", ax = None, show:bool = True):
    '''
        A method for plotting the Loss while training,
        @params:
            loss:list = [] : array that keeps track of loss
            label:str = "Loss" : the label to be displayed in the 
            ax = None: ax that figure os displayed on
            show:bool = True
    '''
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
    
def plotAcc(TrainAcc:list = [],TestAcc:list = [], ax = None, show:bool = True):
    '''
        A method for plotting the Accuracy of test set and training set,
        @params:
            TrainAcc:list = [] : The list that tracks train accuracy
            TestAcc:list = [] : The list that tracks test accuracy
            ax = None: ax that figure os displayed on
            show:bool = True
    '''

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

def Prettysummary(model, dims):
    summary(model,dims)
    
    
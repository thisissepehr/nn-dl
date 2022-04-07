from tqdm import tqdm
from torch import nn
import torch
from torch.optim import Adam,SGD,Adagrad
from utils.evaluate import evaluate

class Train:
    def __init__(self,num_epochs:int = 20, device: torch.device = None, showPlots:bool = True, showLog:bool = True ):
        '''
            Trainer Class with some config parameters
            @params:
                num_epochs:int = 20 : number of epochs that training continues
                device: torch.device = None : device to run on if it is defined previously
                showPlots:bool = True : to show the plots or not
                showLog:bool = True : to print the log while training or not
        '''
        self.max_epochs = num_epochs
        self.loss_epoch = 0
        self.train_accuracy = []
        self.test_accuracy = []
        self.loss_epoch_array = []
        self.showLog = showLog
        self.showPlots = showPlots
        if device is not None : 
            self.device = device
        else : 
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
    def __call__(self, trainLoader:torch.utils.data.DataLoader, testLoader:torch.utils.data.DataLoader, network:nn.Module, lossFunction:torch.nn, optimizer:torch.optim):
        '''
            The caller method of trainer.
            @params: 
                trainLoader:torch.utils.data.DataLoader : the training set data loader
                testLoader:torch.utils.data.DataLoader : the test set data loader
                network:nn.Module : the network module that has the architecture
                lossFunction:torch.nn : the corresponding loss function
                optimizer:torch.optim : the optimization algorithm to be used in training process
        '''
        
        print("notice: training on "+ str(self.device)+"!")
        network.to(self.device)
        for epoch in range(self.max_epochs):
            self.loss_epoch = 0
            for i, data in tqdm(enumerate(trainLoader, 0), ascii=True):
                network.train()
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = network(inputs)
                losses = lossFunction(outputs, labels)
                losses.backward()
                optimizer.step()
                self.loss_epoch += losses.item()
            self.loss_epoch_array.append(self.loss_epoch)
            self.train_accuracy.append(evaluate(network,trainLoader,device=self.device))
            self.test_accuracy.append(evaluate(network,testLoader, device=self.device))
            if self.showLog:
                print("Epoch {}: loss: {}, train accuracy: {}, test accuracy:{}".format(epoch + 1, self.loss_epoch_array[-1], self.train_accuracy[-1], self.test_accuracy[-1]))
        if self.showPlots:
            from utils.plots import plotAcc, plotLoss
            plotAcc(self.train_accuracy,self.test_accuracy)
            plotLoss(self.loss_epoch_array)
                
    def getStats(self):
        return self.train_accuracy,self.test_accuracy
    
    def setClear(self):
        self.loss_epoch = 0
        self.train_accuracy = []
        self.test_accuracy = []
        self.loss_epoch_array = []
        
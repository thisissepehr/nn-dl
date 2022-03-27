from collections import OrderedDict
from sympy import Order
import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
    
    def forward(self,x):
        pass

# TODO : dimensions are not clear
class MLP4(BaseModel):
    def __init__(self,numInputs, numOutputs):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(numInputs,500),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500,250),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(250,250),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(250,numOutputs),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
    
    
class MLPCustom(BaseModel):
    def __init__(self,numLinearLayers:int = 4,dimensions:list = [], dropouts:list = [], activations:list = []  ):
        super().__init__()
        assert len(dropouts) == len(activations) == len(dimensions)
        self.Odict = OrderedDict()
        for i in range(numLinearLayers):
            
            if i!=0 and dimensions[i][0]!=dimensions[i-1][1]:
                raise "Dimensions are not compatible!"
            self.Odict['linear'+str(i)] = nn.Linear(dimensions[i][0],dimensions[i][1])
            if activations[i] == 'Leakyrelu':
                active = nn.LeakyReLU()
            elif activations[i] == 'relu':
                active = nn.ReLU()
            elif activations[i] == 'tanh':
                active = nn.Tanh()
            else:
                active = nn.Sigmoid()
            self.Odict['activation'+str(i)] = active
            if dropouts[i] is not None:
                drop = nn.Dropout(dropouts[i])
                self.Odict['Dropout'+str(i)] = drop
        self.main = nn.Sequential(
            self.Odict
        )
    def forward(self,x):
        return self.main(x)
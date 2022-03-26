import torch
from torch import dropout, nn
import torchvision


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
    
    def forward(self,x):
        pass

# TODO : dimensions are not clear
class MLP4(BaseModel):
    def __init__(self,numInputsm, numOutputs):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784,500),
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
    pass

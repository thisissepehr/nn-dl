from collections import OrderedDict
import torch
from torch import nn, softmax
from models.mlp import BaseModel

class CNN_A(BaseModel):   
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
class CNN_B(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv_Sequence = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(3,3), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
            
            
        )
        self.Linears = nn.Sequential(
            nn.Linear(5824 ,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,10),
            nn.Softmax()
            
        )
        
        
    def forward(self, x):
        x = self.conv_Sequence(x)
        x = x = x.view(x.size(0), -1)
        x = self.Linears(x)
        
        return x
    
class CNN_C(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv_Sequence = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(3,3), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.Linears = nn.Sequential(
            nn.Linear(168 ,100),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100 ,150),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(150,10),
            
        )
        
        
    def forward(self, x):
        x = self.conv_Sequence(x)
        x = x = x.view(x.size(0), -1)
        x = self.Linears(x)
        
        return x
    
from collections import OrderedDict
from turtle import forward
from grpc import insecure_channel
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
    
    
class VGGBlock(BaseModel):
    def __init__(self, numConvs, inputChannels, outputChannels):
        super().__init__()
        self.numConvs = numConvs
        for i in range(self.numConvs):
            self.add_module('conv{0}'.format(i), nn.Conv2d(inputChannels,
                                                           outputChannels,
                                                           kernel_size=3,
                                                           padding=1))
            inputChannels = outputChannels
            self.add_module('relu{0}'.format(i), nn.ReLU())

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        
    def forward(self, x):
        out = x
        for i in range(self.numConvs):
            out = self._modules['conv{0}'.format(i)](out)
            out = self._modules['relu{0}'.format(i)](out)  
        out = self.maxPool(out)
        return out
class VGG11(BaseModel):
    
    def __init__(self,convArch):
        super().__init__()
        self.convArch = convArch
        inChannels = 1
        for i,(numConvs, outChannels) in enumerate(convArch):
            self.add_module('vggBlock{0}'.format(i), VGGBlock(numConvs,
                                                              inChannels,
                                                              outChannels))
            inChannels = outChannels
        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(outChannels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        ) 
        
    def forward(self, x):
        out = x
        for i in range(len(self.convArch)):
            out = self._modules['vggBlock{0}'.format(i)](out)
        out = self.last(out)
        return out
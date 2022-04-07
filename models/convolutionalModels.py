from torch import nn
from models.mlp import BaseModel

class CNN_A(BaseModel):   
    def __init__(self):
        '''
            A CNN architecture with not so many feature maps and batch normalization and maxpooling
        '''
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
        '''
            A CNN with only on convolutional and two linear layers and using maxpooling
        '''
        
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
        '''
            A more sophisticated CNN with two convolutional layers with maxpooling and three linears with dropouts 
        '''
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
    
    
class VGG11(BaseModel):
    def __init__(self, in_channels, num_classes=10):
        '''
            an implementation of VGG11 but it still needs work to be functional!
        '''
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
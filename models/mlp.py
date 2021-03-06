from collections import OrderedDict
import torch
from torch import nn


class BaseModel(nn.Module):
    '''
        This is the Parent model for every model created here The reason is to lose the pytorch signature passing
    '''
    def __init__(self):
        super(BaseModel,self).__init__()
    
    def forward(self,x):
        pass

class STEM(BaseModel):
    def __init__(self, numPixelsTopatch, img_height:int = 28, img_width:int = 28):
        '''
            The STEM Layer for the assigned model
            @params:
                numPixelsTopatch: the window size for patches
                img_height:int = 28 : height of the original input image
                img_width:int = 28 : width of the original input image
        '''
        super().__init__()
        self.numPixelsTopatch = numPixelsTopatch
        self.finalLinear = nn.Linear((img_height*img_width)//((img_height//numPixelsTopatch) * (img_width//numPixelsTopatch)),784)
    def __patching(self,batch):
        final = []
        batch = batch.unfold(2, self.numPixelsTopatch, self.numPixelsTopatch).unfold(3, self.numPixelsTopatch, self.numPixelsTopatch)
        for img in batch:
          p = img.flatten()
          p = torch.split(p,196)
          p = torch.stack(list(p), dim=0)
          final.append(p)
        final = torch.stack(final)
        return final
        
    def forward(self,x):
    
        assert x.size()[2] * x.size()[3] // self.numPixelsTopatch != 0
        x = self.__patching(x)
        x = self.finalLinear(x)
        return x
        
        


class BackBoneBlock(BaseModel):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

    
class ClassifierCell(BaseModel):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

class MLP4(BaseModel):
    def __init__(self,numInputs, numOutputs, patching:bool = True):
        '''
            The 4 Layer MLP
            @params:
                numInputs: the size of the input, the number of hidden units to process them in the first Linear layer
                numOutputs: the number of expected outputs, the number of classes
                patching: to patch or not
        '''
        super().__init__()
        self.numInputs = numInputs
        self.doPatch = patching
        if self.doPatch:
            self.StemPatching = STEM(14)
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
        )
    def forward(self, x):
        if self.doPatch:
            x = self.StemPatching(x)
        else:
            x = x.view(-1, self.numInputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.doPatch:
            x = x.mean(axis=1)
        x = self.fc4(x)
        return x
    
class MeanLayer(BaseModel):
    '''
        A pytoch style layer just to get the mean of the patches
    '''
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        return x.mean(axis=1)
 
    
class MLPCustom(BaseModel):
    def __init__(self,numLinearLayers:int = 4,dimensions:list = [], dropouts:list = [], activations:list = [], patching:bool = False):
        '''
            A model that can be used as a customizer for MLPs.
            @params:
                numLinearLayers:int = 4, number of Linear layers expected to be in the model architecture
                dimensions:list = [], the hidden units in each linear layer, Dimensions must match
                dropouts:list = [], dropout rate on each Linear model
                activations:list = [], activation functions after each layer
                patching:bool = False, do patching or not
        '''
        super().__init__()
        assert len(dropouts) == len(activations) == len(dimensions) == numLinearLayers
        self.numInputs = dimensions[0][0]
        self.Odict = OrderedDict()
        self.doPatch = patching
        
        for i in range(numLinearLayers):
            
            if i!=0 and dimensions[i][0]!=dimensions[i-1][1]:
                raise "Dimensions are not compatible!"
            if i==0 and self.doPatch:
                self.Odict["Stem"+str(i)] = STEM(14)
            if i==numLinearLayers-1 and self.doPatch:
                self.Odict['Mean'] = MeanLayer()
            self.Odict['linear'+str(i)] = nn.Linear(dimensions[i][0],dimensions[i][1])
            if activations[i] == 'Leakyrelu':
                active = nn.LeakyReLU()
            elif activations[i] == 'relu':
                active = nn.ReLU()
            elif activations[i] == 'tanh':
                active = nn.Tanh()
            else:
                active = nn.Sigmoid()
            if i!=numLinearLayers-1:    
                self.Odict['activation'+str(i)] = active
                if dropouts[i] is not None:
                    drop = nn.Dropout(dropouts[i])
                    self.Odict['Dropout'+str(i)] = drop
        self.main = nn.Sequential(
            self.Odict
        )
    def forward(self,x):
        if not self.doPatch:
            x = x.view(-1, self.numInputs)
        return self.main(x)

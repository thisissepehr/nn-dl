from collections import OrderedDict
import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
    
    def forward(self,x):
        pass

class STEM(BaseModel):
    def __init__(self, numPatches):
        super().__init__()
        self.numPatches = numPatches
    
    def __patching(self,batch):
        final = []
        batch = batch.unfold(2, self.numPatches, self.numPatches).unfold(3, self.numPatches, self.numPatches)
        for img in batch:
          p = img.flatten()
          p = torch.split(p,196)
          p = torch.stack(list(p), dim=0)
          final.append(p)
        final = torch.stack(final)
        return final
        
    def forward(self,x):
    
        assert x.size()[2] * x.size()[3] // self.numPatches != 0
        return self.__patching(x)
        
        


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
    def __init__(self,numInputs, numOutputs):
        super().__init__()
        self.numInputs = numInputs
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
        self.testing = STEM(14)
    def forward(self, x):
        # x = x.view(-1, self.numInputs)
        x = self.testing(x)
        # TODO : size consistency here is a key :)
        print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
    
    
class MLPCustom(BaseModel):
    def __init__(self,numLinearLayers:int = 4,dimensions:list = [], dropouts:list = [], activations:list = []):
        super().__init__()
        assert len(dropouts) == len(activations) == len(dimensions) == numLinearLayers
        self.numInputs = dimensions[0][0]
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
            if i!=numLinearLayers-1:    
                self.Odict['activation'+str(i)] = active
                if dropouts[i] is not None:
                    drop = nn.Dropout(dropouts[i])
                    self.Odict['Dropout'+str(i)] = drop
        self.main = nn.Sequential(
            self.Odict
        )
    def forward(self,x):
        x = x.view(-1, self.numInputs)
        return self.main(x)

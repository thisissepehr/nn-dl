import torch

def init_weights(m):
    if type(m) == torch.nn.Linear: # by checking type we can init different layers in different ways
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.zeros_(m.bias)
        
        
def Xavier(m):
    pass
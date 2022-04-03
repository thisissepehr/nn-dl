import torch
from torchvision import transforms
from models.mlp import BaseModel

class AutoEncoder(BaseModel):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass
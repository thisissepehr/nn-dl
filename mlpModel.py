from loader.dataloader import FashionMNISTdataHandler
from models import mlp


trainLoader, testLoader = FashionMNISTdataHandler("./FashionMNIST")()


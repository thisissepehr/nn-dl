from loader.dataloader import FashionMNISTdataHandler
from models import mlp, convolutionalModels
from utils.initializers import Normal, Xavier, KaimingHe, custom_init
from torch import nn
from torch.optim import Adam,SGD,Adagrad
from utils.train import Train

trainLoader, testLoader = FashionMNISTdataHandler("./FashionMNIST")(batchSize=32)
# //////////////////////////////////
# //       network Defining       //
# //////////////////////////////////


# /////////////// block one ////////
# net = mlp.MLP4(784,10, patching = True)
# loss = nn.CrossEntropyLoss()
# lr = 0.1
# weightDecay = 0
# optimizer = SGD(net.parameters(), lr =lr, weight_decay=weightDecay)
# net.apply(Normal)
# print(net)

# reaches 88.44% as test accuracy without patching, with patching 86.92
# //////////////////////////////////



#///////////// block two //////////////
# net = mlp.MLPCustom(activations=["Leakyrelu","Leakyrelu","Leakyrelu","Leakyrelu","Leakyrelu"],
#                     dimensions=[(784,900),(900,500),(500,450),(450,250),(250,10)],
#                     numLinearLayers=5,
#                     dropouts=[0.3,0.3,0.3,0.3,0.3],
#                     patching=False)
# loss = nn.CrossEntropyLoss()
# lr = 0.1
# weightDecay = 0
# momentum = 0.9
# optimizer = SGD(net.parameters(), lr =lr, weight_decay=weightDecay, momentum = momentum)
# net.apply(Normal)
# print(net)
# this model did 88.87 on test dataset
#///////////////////////////////

#////////////////// block three ///////////////////
# net = convolutionalModels.CNN_A()
# lr = 0.07
# optimizer = Adam(net.parameters(), lr= 0.07)
# loss = nn.CrossEntropyLoss()
# net.apply(Normal)
# print(net)
# this reached 86.05 on test
#//////////////////////////////////////////////////


#////////////////// block four  ///////////////////
# net = convolutionalModels.CNN_B()
# lr = 0.01
# momentum = 0.9
# # optimizer = Adam(net.parameters(), lr= 0.04)
# optimizer = SGD(net.parameters(), lr = lr, momentum= momentum)
# loss = nn.CrossEntropyLoss()
# net.apply(custom_init)
# print(net)
# this reached 83.9 on test
#//////////////////////////////////////////////////

#////////////////// block five  ///////////////////
# batchsize = 32
# net = convolutionalModels.CNN_C()
# lr = 0.015
# wd=0.9
# momentum = 0.9
# optimizer = SGD(net.parameters(), lr = lr, momentum= momentum)
# loss = nn.CrossEntropyLoss()
# net.apply(custom_init)
# print(net)
# this reached 90.62 on test
#//////////////////////////////////////////////////

#////////////////// block six   ////////////////////
batchsize = 32
convArch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = convolutionalModels.VGG11(in_channels=1, num_classes=10)
lr = 0.015
wd=0.9
momentum = 0.9
optimizer = Adam(net.parameters(), lr = lr)#, momentum= momentum)
loss = nn.CrossEntropyLoss()
net.apply(custom_init)
print(net)
# this architecture doesnt work on my laptop :(
# this reached ? on test
#//////////////////////////////////////////////////







# //////////////////////////////////
# //         training             //
# //////////////////////////////////
t = Train()(trainLoader=trainLoader,
            testLoader=testLoader,
            network=net,
            lossFunction=loss,
            optimizer=optimizer,
            )

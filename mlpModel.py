import torch
from loader.dataloader import FashionMNISTdataHandler
from models import mlp, convolutionalModels
from utils.evaluate import evaluate
from utils.plots import plotAcc, plotLoss
from utils.initializers import Normal, Xavier, KaimingHe, custom_init
from torch import nn
from torch.optim import Adam,SGD,Adagrad
from tqdm import tqdm



trainLoader, testLoader = FashionMNISTdataHandler("./FashionMNIST")(batchSize=32)
# //////////////////////////////////
# //       network Defining       //
# //////////////////////////////////


# /////////////// block one ////////
# net = mlp.MLP4(784,10)
# loss = nn.CrossEntropyLoss()
# lr = 0.1
# weightDecay = 0
# optimizer = SGD(net.parameters(), lr =lr, weight_decay=weightDecay)
# net.apply(Normal)
# print(net)

# reaches 88.44% as test accuracy
# //////////////////////////////////



#///////////// block two //////////////
# net = mlp.MLPCustom(activations=["Leakyrelu","Leakyrelu","Leakyrelu","Leakyrelu","Leakyrelu"],
#                     dimensions=[(784,900),(900,500),(500,450),(450,250),(250,10)],
#                     numLinearLayers=5,
#                     dropouts=[0.3,0.3,0.3,0.3,0.3])
# loss = nn.CrossEntropyLoss()
# lr = 0.1
# weightDecay = 0
# optimizer = SGD(net.parameters(), lr =lr, weight_decay=weightDecay)
# net.apply(Normal)
# print(net)
# this model did 87.51 on test dataset
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
net = convolutionalModels.CNN_C()
lr = 0.015
wd=0.9
momentum = 0.9
optimizer = SGD(net.parameters(), lr = lr, momentum= momentum)
loss = nn.CrossEntropyLoss()
net.apply(custom_init)
print(net)
# this reached 90.62 on test
#//////////////////////////////////////////////////







# //////////////////////////////////
# //         training             //
# //////////////////////////////////


loss_epoch_array = []
max_epochs = 20
loss_epoch = 0
train_accuracy = []
test_accuracy = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(max_epochs):
    loss_epoch = 0
    for i, data in tqdm(enumerate(trainLoader, 0), ascii=True):
        net.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        losses = loss(outputs, labels)
        losses.backward()
        optimizer.step()
        loss_epoch += losses.item()
    loss_epoch_array.append(loss_epoch)
    train_accuracy.append(evaluate(net,trainLoader,device=device))
    test_accuracy.append(evaluate(net,testLoader, device=device))
    print("Epoch {}: loss: {}, train accuracy: {}, test accuracy:{}".format(epoch + 1, loss_epoch_array[-1], train_accuracy[-1], test_accuracy[-1]))

plotAcc(train_accuracy,test_accuracy)
plotLoss(loss_epoch_array)
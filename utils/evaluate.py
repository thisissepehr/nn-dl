import torch

def evaluate(net, dataloader, device):
    '''
        a method for evaluation of the updates to the netork,
        @params:
            net: network module
            dataloader: the dataset that you want to do the evaluation on 
            device: the device that all the computation is being processed on
    '''
    total, correct = 0,0
    net.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total

import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cPickle as pickle
import numpy
import argparse

from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
        help='input batch size for training (default: 100)')
args = parser.parse_args()

print('=> loading cifar100 data...')
normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
        
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ]))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifer = nn.Sequential(
                nn.Dropout(0.2),
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.Conv2d(96,  96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192,  100, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifer(x)
        x = x.view(x.size(0), 100)
        return x


model = Net()
print model
model.cuda()
model = torch.nn.DataParallel(model)

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.05)
        m.bias.data.normal_(0.0, 0.01)



criterion = nn.CrossEntropyLoss()
param_dict = dict(model.named_parameters())
params = []

base_lr = 0.05

for key, value in param_dict.items():
    params += [{'params':[value], 'lr':base_lr,
        'momentum':0.9, 'weight_decay':0.0005}]


optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.001)

def train(epoch):
    model.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_avg += loss
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0],
                optimizer.param_groups[1]['lr']))
    loss_avg /= len(trainloader.dataset)
    print('Avg train loss: {:.4f}'.format(loss_avg * 128))
            

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 128., float(correct), len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))

def adjust_learning_rate(optimizer, epoch):
    S = [200, 250, 300]
    if epoch in S:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def print_std():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            print torch.std(m.weight.data)

for epoch in range(350):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test()

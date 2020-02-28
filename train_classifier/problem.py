import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import torch.utils.data as td
from torch.utils.data.dataset import Dataset 
import math
import os

# define customized dataset (using generated data)
class CustomDataset(Dataset):
    def __init__(self, path, height, width, channels, transform=None):
        self.data = np.load(path)['sample']
        self.labels = np.load(path)['label']
        self.height = height
        self.width = width
        self.channels = channels
        self.transform = transform

    def __getitem__(self, index):
        img_as_np = self.data[index].reshape(self.channels, self.height, self.width)
        single_image_label = self.labels[index]

        # transform image to tensor
        img_as_tensor = torch.from_numpy(img_as_np).float()
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_tensor)
        else:
            img_as_tensor = torch.from_numpy(img_as_np).float()

        # return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)

## standard MNIST and CIFAR-10 loaders
def mnist_loaders(batch_size, path, is_shuffle=False): 
    mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())

    train_loader = td.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(mnist_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    return train_loader, test_loader

def cifar_loaders(batch_size, path, is_shuffle=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_train = datasets.CIFAR10(path, train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    cifar_test = datasets.CIFAR10(path, train=False, 
                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_loader = td.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(cifar_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    return train_loader, test_loader

## custom MNIST and CIFAR-10 loaders
def custom_mnist_loaders(batch_size, path, is_shuffle=False):
    mnist = CustomDataset(path, 28, 28, 1)
    loader = td.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    return loader       

def custom_cifar_loaders(batch_size, path, is_shuffle=False): 
    imagenet10 = CustomDataset(path, 32, 32, 3)
    loader = td.DataLoader(imagenet10, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    return loader

## MNIST model (zico)
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

## MNIST model (madry, trades)
class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

## CIFAR-10 Model (madry, trades)
class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, input):
        logits = self.classifier(input)
        return logits
 
## CIFAR-10 models (zico)
def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
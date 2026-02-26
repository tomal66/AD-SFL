import torch
import torch.nn as nn
from torchvision.models import resnet18, wide_resnet50_2

class ResNet18Client(nn.Module):
    def __init__(self, dataset="CIFAR10"):
        super().__init__()
        resnet = resnet18(weights=None)
        
        # for CIFAR10/100, the first conv is usually replaced to handle 32x32 images
        if dataset in ["CIFAR10", "CIFAR100"]:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
            
        self.client_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # Identity for CIFAR
            resnet.layer1
        )
        
    def forward(self, x):
        return self.client_layers(x)

class ResNet18Server(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        resnet = resnet18(weights=None)
        
        self.server_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.server_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class WideResNet50Client(nn.Module):
    def __init__(self, dataset="CIFAR100"):
        super().__init__()
        resnet = wide_resnet50_2(weights=None)
        
        if dataset in ["CIFAR10", "CIFAR100"]:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
            
        self.client_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        
    def forward(self, x):
        return self.client_layers(x)

class WideResNet50Server(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        resnet = wide_resnet50_2(weights=None)
        
        self.server_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.server_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

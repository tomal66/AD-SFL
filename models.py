
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MNIST Models (Simple CNN) ---

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Output after pool: 64 x 14 x 14

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # Keep feature map for backbone
        return x

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Input: 64 x 14 x 14
        # After conv3: 128 x 14 x 14
        # After pool: 128 x 7 x 7

    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

class Tail(nn.Module):
    def __init__(self, num_classes=10):
        super(Tail, self).__init__()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- CIFAR-10 Models (ResNet-18 Split) ---

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetHead(nn.Module):
    """
    Client-side Head for ResNet-18.
    Contains initial conv, bn, layer1.
    Output: 64 x 32 x 32 (stride 1 in layer1)
    """
    def __init__(self):
        super(ResNetHead, self).__init__()
        self.in_planes = 64
        # CIFAR-10 input is 3x32x32.
        # Standard ResNet uses 7x7 conv stride 2, but for CIFAR usually 3x3 s1 is used to keep dims.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out

class ResNetBackbone(nn.Module):
    """
    Server-side Backbone for ResNet-18.
    Contains layer2, layer3.
    Input: 64 x 32 x 32
    layer2 (stride 2) -> 128 x 16 x 16
    layer3 (stride 2) -> 256 x 8 x 8
    Output: 256 x 8 x 8
    """
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64 # Output of layer1
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        return out

class ResNetTail(nn.Module):
    """
    Client-side Tail for ResNet-18 (or Server, depending on split setting, but standard SL puts tail on server usually?
    SafeSplit implies Client(Head) -> Server(Backbone) -> Client(Tail) for labeling?
    Actually SafeSplit paper: Client1 (Head+Tail) <-> Server (Backbone).
    Wait, original code puts Head and Tail on Client.
    
    ResNetTail:
    Contains layer4, linear.
    Input: 256 x 8 x 8
    layer4 (stride 2) -> 512 x 4 x 4
    avgpool -> 512 x 1 x 1
    linear -> 10
    """
    def __init__(self, num_classes=10):
        super(ResNetTail, self).__init__()
        self.in_planes = 256 # Output of layer3
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- Factory ---

def get_models(dataset_name, device):
    if dataset_name == 'mnist':
        return Head().to(device), Backbone().to(device), Tail().to(device)
    elif dataset_name == 'cifar10':
        return ResNetHead().to(device), ResNetBackbone().to(device), ResNetTail().to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

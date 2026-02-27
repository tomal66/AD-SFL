import torch
import torch.nn as nn
from torchvision.models import resnet18, wide_resnet50_2, Wide_ResNet50_2_Weights

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

# -----------------------
# WideResNet50_2 (FIXED for Split Learning)
# -----------------------
def build_wide_resnet50_2_backbone(dataset="CIFAR100", weights="DEFAULT"):
    if weights == "DEFAULT":
        weights = Wide_ResNet50_2_Weights.DEFAULT
    elif weights is None or weights == "NONE":
        weights = None

    backbone = wide_resnet50_2(weights=weights)

    if dataset in ["CIFAR10", "CIFAR100"]:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

    return backbone


class WideResNet50Client(nn.Module):
    """
    Client takes a shared backbone instance.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.client_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )

    def forward(self, x):
        return self.client_layers(x)


class WideResNet50Server(nn.Module):
    """
    Server takes the SAME shared backbone instance and attaches a paper-aligned classifier:
    just one Linear layer (replace final layer only).
    """
    def __init__(self, backbone: nn.Module, num_classes=100):
        super().__init__()
        self.server_layers = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )

        in_ftrs = backbone.fc.in_features  # 2048 for wide_resnet50_2
        self.fc = nn.Linear(in_ftrs, num_classes)

    def forward(self, x):
        x = self.server_layers(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_wideresnet50_split(dataset="CIFAR100", num_classes=100, weights="DEFAULT"):
    """
    Convenience helper: returns (client, server) built from ONE shared backbone.
    """
    backbone = build_wide_resnet50_2_backbone(dataset=dataset, weights=weights)
    client = WideResNet50Client(backbone)
    server = WideResNet50Server(backbone, num_classes=num_classes)
    return client, server
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClientModel(nn.Module):
    """
    The client-side portion of the split model.
    Contains the first few layers up to the 'cut layer'.
    """
    def __init__(self, in_channels=1, hidden_channels=32):
        super(ClientModel, self).__init__()
        # e.g., first conv layer of a small CNN for MNIST
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x

class ServerModel(nn.Module):
    """
    The server-side portion of the split model.
    Contains the remaining layers from the 'cut layer' to the output.
    """
    def __init__(self, in_channels=32, hidden_channels=64, num_classes=10, input_size=(14, 14)):
        super(ServerModel, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        flat_size = hidden_channels * (input_size[0] // 2) * (input_size[1] // 2)
        
        self.fc1 = nn.Linear(flat_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

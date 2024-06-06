# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, input_size=32, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dynamically calculate the size of the fully connected layer input
        self._to_linear = None
        self._calculate_to_linear(input_size)
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_to_linear(self, input_size):
        # Pass a dummy tensor through the convolutional layers to calculate the output size
        x = torch.randn(1, 3, input_size, input_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        self._to_linear = x.numel()
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleResNet(nn.Module):
    
    def __init__(self, input_size=32, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self._to_linear = None
        self._calculate_to_linear(input_size)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # to calculate what the final dimension of the image will be if it passed through the layers and it calculated by sending a dummy tensor through it
    # then it is passed through the network and it was important to know the input dimension for the final fully connected layer.
    def _calculate_to_linear(self, input_size):
        x = torch.randn(1, 3, input_size, input_size)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, 3, padding=1
        )  # input: 3 channels (RGB), output:16 filters
        self.pool = nn.MaxPool2d(2, 2)  # max pooling 2x2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # assuming input images 32x32
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
        x = torch.flatten(x, 1)  # flatten except batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

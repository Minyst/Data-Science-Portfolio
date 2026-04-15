import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        z = F.relu(self.norm2(self.conv2(y)))
        return self.conv3(x) + z

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.block1 = block(64, 128)
        self.block2 = block(128, 256)
        self.block3 = block(256, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 100)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
    
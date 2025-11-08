import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel1=3, kernel2 = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel1, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2=3, padding1=1, padding2=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, padding1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, padding2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCNN4Conv(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, kernel1, 1, 1)
        self.conv4 = nn.Conv2d(64, 32, kernel2, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCNN6Conv(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, kernel1, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, kernel2, 1, 1)
        self.conv5 = nn.Conv2d(128, 64, kernel1, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, kernel2, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2 = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock(32, 32, kernel1, kernel2)
        self.res2 = ResidualBlock(32, 64, 2, kernel1, kernel2)
        self.res3 = ResidualBlock(64, 64, kernel1, kernel2)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, kernel, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, kernel, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
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


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
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

class CNNWithBottleneck(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2 = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = BottleneckBlock(32, 32)
        self.res2 = BottleneckBlock(128, 64)
        self.res3 = BottleneckBlock(256, 64)
        
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


class CNNWithWideres(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, kernel1=3, kernel2 = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = WideResidualBlock(32, 32)
        self.res2 = WideResidualBlock(32, 64, 2)
        self.res3 = WideResidualBlock(64, 64)
        
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


class CastomNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class CastomWithNoiseSimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, kernel1=3, kernel2=3, padding1=1, padding2=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, padding1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, padding2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.noise = CastomNoise(std=0.1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.noise(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        hidden_dim = num_channels // reduction_ratio
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        
        z = self.squeeze(x)
        z = z.view(B, C)
        
        s = self.excitation(z)
        s = s.view(B, C, 1, 1)
        
        return x * s
    
class CastomAttentionSimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, kernel1=3, kernel2=3, padding1=1, padding2=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, padding1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, padding2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.attention = Attention(64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


class CastomGeluSimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, kernel1=3, kernel2=3, padding1=1, padding2=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, padding1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, padding2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.gelu = GELU()
    
    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.pool(self.gelu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CastomSoftPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_exp = torch.exp(x)

        numerator = F.avg_pool2d(
            x * x_exp,
            self.kernel_size,
            self.stride,
            self.padding,
            count_include_pad=True
        )

        denominator = F.avg_pool2d(
            x_exp,
            self.kernel_size,
            self.stride,
            self.padding,
            count_include_pad=True
        )

        return numerator / (denominator + 1e-8)


class CastomPoolSimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, kernel1=3, kernel2=3, padding1=1, padding2=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel1, 1, padding1)
        self.conv2 = nn.Conv2d(32, 64, kernel2, 1, padding2)
        self.pool = CastomSoftPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
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
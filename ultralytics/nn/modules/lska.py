import torch
import torch.nn as nn

class LSKA(nn.Module):
    def __init__(self, channels, kernel_size=15, reduction=16):
        super().__init__()
        padding = kernel_size // 2

        # Depthwise Separable Large Kernel Conv
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=channels)

        # Channel attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        w = self.pool(out)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)

        out = out * w
        return out + x

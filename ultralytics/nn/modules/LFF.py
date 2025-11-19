import torch
import torch.nn as nn
import torch.nn.functional as F

class LFF(nn.Module):
    """
    Light Feature Filter (LFF)
    轻量通道重标定模块，增强小目标特征，抑制背景噪声
    """
    def __init__(self, channels):
        super(LFF, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(channels)  # 可选，可提高训练稳定性
        self.act = nn.Hardsigmoid()         # Hard-Sigmoid 激活

    def forward(self, x):
        # 生成通道权重
        w = self.conv1(x)
        w = self.bn(w)       # 可选
        w = self.act(w)
        # 通道重标定
        out = x * w
        return out

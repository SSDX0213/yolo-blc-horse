import torch
import torch.nn as nn

class LR_ASPP(nn.Module):
    """
    LR-ASPP: Lightweight Residual ASPP
    d=1 and d=3 (or d=6)
    """
    def __init__(self, c):
        super().__init__()
        
        # local branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),  # DWConv
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),  # PWConv
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

        # dilation branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=3, dilation=3, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

        # fuse
        self.fuse = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = self.fuse(torch.cat([b1, b2], dim=1))
        return out + x   # residual addclass LR_ASPP

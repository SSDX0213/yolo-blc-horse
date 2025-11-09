import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    """
    Boundary-Aware Module (BAM)
    可自动支持 YOLO YAML 多输入形式:
    例如: - [[2,4,9], 1, BAM, [64,128,256]]
    """

    def __init__(self, c1=64, c2=128, c3=256, out_c=256):
        super(BAM, self).__init__()
        # 允许 c1/c2/c3 传入 list
        if isinstance(c1, (list, tuple)):
            # 如果 YAML 传的是 [64,128,256]
            c1, c2, c3 = c1
        self.out_c = out_c

        self.conv1 = nn.Conv2d(c1, out_c, 1)
        self.conv2 = nn.Conv2d(c2, out_c, 1)
        self.conv3 = nn.Conv2d(c3, out_c, 1)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.deep_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x 可能是:
        - 一个列表 [f2, f4, f9]
        - 或者单个特征图（为了兼容单输入模式）
        """
        if isinstance(x, (list, tuple)):
            # 自动展开
            f1, f2, f3 = x
        else:
            # 单输入情况 (兼容)
            f1 = f2 = f3 = x

        target_size = f3.shape[-2:]
        f1 = F.interpolate(self.conv1(f1), size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.conv2(f2), size=target_size, mode='bilinear', align_corners=False)
        f3 = self.conv3(f3)

        f12 = self.fuse_conv(f1 + f2)
        f123 = self.deep_conv(f12 * f3) + f12

        avg_map = torch.mean(f123, dim=1, keepdim=True)
        max_map, _ = torch.max(f123, dim=1, keepdim=True)
        att = self.spatial_att(torch.cat([avg_map, max_map], dim=1))

        return f123 * att





# 目标分离注意力模块（OSA）
class OSA(nn.Module):
    def __init__(self, channels):
        super(OSA, self).__init__()
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxx, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg, maxx], dim=1))
        channel = self.channel_att(x)
        return x * spatial * channel

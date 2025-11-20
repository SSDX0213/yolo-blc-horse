import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 1. PConv (通道保持) -----------------
class PConv_Refined(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.channels = channels
        
        # 1. PConv 只对输入通道的1/4进行卷积
        self.conv_channels = channels // 4
        
        # 2. 关键修正：如果输入通道太少，至少让 1 个通道参与计算
        if self.conv_channels == 0:
            self.conv_channels = 1
            
        # 3. 确保实际卷积的输入和输出通道数 > 0
        self.conv = nn.Conv2d(self.conv_channels, self.conv_channels, 
                              kernel_size, stride, padding, groups=groups, bias=bias)
    def forward(self, x):
        # 1. 通道分割
        # x.shape[1] 确保在 in_channels 无法被 4 整除时，能正确计算非计算通道数
        x1, x2 = torch.split(x, [self.conv_channels, x.shape[1] - self.conv_channels], dim=1)
        
        # 2. 卷积计算
        x1 = self.conv(x1)
        
        # 3. 通道拼接 (输出通道与输入通道相同)
        return torch.cat((x1, x2), dim=1)

    
    
    
    # ----------------- 2. PCC_Bottleneck (集成 PConv 和 CA) -----------------
# 假设 CoordAtt (CA) 模块已按照我之前的代码实现
class PCC_Bottleneck(nn.Module):
    """
    PConv-CA-Bottleneck (PCC-Bottleneck):
    c1 -> c_ (1x1) -> PConv (c_ -> c_) -> cv2 (1x1, c_ -> c2) -> CA
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (通道缩减后的中间通道数)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)  # 第一个 1x1 卷积: c1 -> c_
        
        # 核心替换：使用通道保持的 PConv_Refined (c_ -> c_)
        self.pconv = PConv_Refined(c_, 3, 1, 1, groups=g) 
        
        # 第二个 1x1 卷积: c_ -> c2 (负责通道恢复/调整)
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, 0, bias=False) 

        self.add = shortcut and c1 == c2
        self.ca = CoordAtt(c2, c2) # CA 模块集成在输出通道 c2 上

    def forward(self, x):
        # 1. 1x1 Conv (通道缩减)
        y = self.cv1(x)
        
        # 2. PConv (轻量化 3x3 操作)
        y = self.pconv(y) 
        
        # 3. 1x1 Conv (通道恢复)
        out = self.cv2(y)
        
        # 4. CA (通道增强)
        out = self.ca(out)
        
        return x + out if self.add else out
    
    
    
    
# ----------------- 2. CA (Coordinate Attention) 模块 -----------------
class CoordAtt(nn.Module):
    """
    Coordinate Attention (CA) 模块
    通过两个一维池化捕获水平和垂直方向的位置信息。
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 降维后的中间通道数
        mid_channels = max(8, in_channels // reduction)
        
        # 共享 1x1 卷积层 (用于通道降维和融合)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish() # 沿用 YOLOv8 的激活函数

        # 独立 1x1 卷积层 (用于学习权重)
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        N, C, H, W = x.size()

        # 1. 水平/垂直池化
        x_h = self.pool_h(x) # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (N, C, 1, W) -> (N, C, W, 1) -> permute to (N, C, 1, W) for cat

        # 2. 拼接和降维 (用于信息交互)
        x_cat = torch.cat([x_h, x_w], dim=2) # (N, C, H+W, 1)

        # 3. 共享卷积和激活
        out = self.act(self.bn1(self.conv1(x_cat))) # (N, mid_channels, H+W, 1)
        
        # 4. 通道分离
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # (N, mid_channels, 1, W)

        # 5. 学习权重
        a_h = self.conv_h(x_h).sigmoid() # (N, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid() # (N, C, 1, W)

        # 6. 加权
        return identity * a_w * a_h
    
    
    
    
# ----------------- 3. PCC-Block (C2f) 结构 (不变) -----------------
class PCC_C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        
        # print(f"PCC_C2f INIT: c2 received = {c2}") # 检查 c2 是否等于 32
        
        # 1. 计算隐藏通道 self.c
        self.c = int(c2 * e)  # hidden channels
        
        # 2. 关键修正：设置最小通道数，防止 c=0
        if self.c == 0:
            self.c = 1 # 保证通道数至少为1
            
        # 3. 确保 cv1 的输出通道 (2 * self.c) > 0
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, 0, bias=False) 
        
        # 4. 确保 cv2 的输入通道 > 0
        self.cv2 = nn.Conv2d(int((2 + n) * self.c), c2, 1, 1, 0, bias=False) 
        
        # 5. Bottleneck 列表的输入 self.c 也大于 0
        self.m = nn.ModuleList(
            PCC_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1)) 
        y.extend(m(y[-1]) for m in self.m)

        # 计算并赋值给 output_tensor
        output_tensor = self.cv2(torch.cat(y, 1))
        # print(f"PCC_C2f Output Shape: {output_tensor.shape}")

        # 仅返回计算好的张量
        return output_tensor
        
        return self.cv2(torch.cat(y, 1)) # 通道拼接和融合
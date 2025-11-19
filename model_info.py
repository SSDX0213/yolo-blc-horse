from ultralytics import YOLO
from torchinfo import summary
import torch
from thop import profile

model = YOLO("yolov8-seg.yaml").model

# 统计参数量
summary(model, input_size=(1, 3, 640, 640))

model.cpu()  # 强制模型搬到CPU

# 统计FLOPs（MACs），注意输入必须一致
dummy = torch.randn(1, 3, 640, 640)
macs, params = profile(model, inputs=(dummy,))
print("MACs:", macs/1e9, "GFLOPs")
print("Params:", params/1e6, "M")


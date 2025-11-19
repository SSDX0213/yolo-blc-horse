from ultralytics import YOLO
import torch
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1️⃣ 加载 YOLOv11s-seg 模型结构
    model = YOLO("yolo11s-seg.yaml")   # ✅ 改这里

    # 2️⃣ 可选：加载预训练权重
    model.load("yolo11s-seg.pt")       # ✅ 改这里

    # 3️⃣ 开始训练
    model.train(
        data="./horse.yaml",             # 你的数据配置文件
        epochs=300,
        imgsz=640,
        batch=32,
        project="horse_cascade_models",
        name="Horse/orin_v11s",  # ✅ 建议修改名字防混淆
        device="0",
        lr0=0.002,
        lrf=0.01
    )

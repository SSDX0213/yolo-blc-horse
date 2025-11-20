from ultralytics import YOLO
import torch
import os



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = YOLO("./yolov8-seg.yaml")

    # 3. 训练（默认激活函数）
    model.train(
        data="./horse.yaml",  # 你的数据配置文件
        epochs=200,
        imgsz=640,
        batch=64,
        project="horse_models_lite",
        name="Orin",
        device="0",
        lr0=0.002,
        lrf=0.01,
        cache="ram"
    )

from ultralytics import YOLO
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = YOLO("./orin_yolov8n-seg.yaml")

    # 2. 可选：加载预训练权重（推荐）
    model.load("./yolov8n-seg.pt")

    # 3. 训练（默认激活函数）
    model.train(
        data="./horse.yaml",  # 你的数据配置文件
        epochs=300,
        imgsz=640,
        batch=32,
        project="horse_cascade_models",
        name="Horse/True_LSKA_BAM_23",
        device="0",
        lr0=0.002,
        lrf=0.01
    )

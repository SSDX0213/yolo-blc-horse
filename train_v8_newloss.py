from ultralytics import YOLO
from loss_trainer import MySegTrainer

model = YOLO("yolov8-seg-PCC.yaml")

model.train(
    data="./part.yaml",  # 你的数据配置文件
    epochs=200,
    imgsz=640,
    batch=64,
    project="horse_models_lite_all",
    name="PCC_part",
    device="0",
    lr0=0.002,
    lrf=0.01,
    cache="ram",
    trainer=MySegTrainer
)

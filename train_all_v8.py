import os
from ultralytics import YOLO
from loss_trainer import MySegTrainer

BASE = "/root/autodl-tmp/photoslabels"

# 训练参数统一配置
TRAIN_ARGS = dict(
    epochs=200,
    imgsz=640,
    batch=128,
    device="0",
    lr0=0.002,
    lrf=0.01,
    cache="ram",
    trainer=MySegTrainer
)

def main():
    # 遍历 photoslabels 下的所有文件夹
    folders = [
        f for f in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, f))
    ]

    print("将训练以下任务：", folders)

    for folder in folders:
        folder_path = os.path.join(BASE, folder)

        # YAML 配置文件必须与文件夹同名
        yaml_path = os.path.join(f"./{folder}.yaml")
        if not os.path.exists(yaml_path):
            print(f"跳过 {folder}：未找到 {folder}.yaml")
            continue
#         if folder == "body_feature":
#             print("训练过，跳过")
#             continue
    
        print(f"\n==============================")
        print(f"开始训练：{folder}")
        print(f"使用 YAML：{yaml_path}")
        print("==============================\n")

        # 每个任务一个模型实例
        model = YOLO("yolov8-seg-PCC.yaml")

        model.train(
            data=yaml_path,
            project="horse_models_lite_all",
            name="orin_"+folder,  # 保存为不同名字
            **TRAIN_ARGS
        )

    print("\n全部任务训练完成。")

if __name__ == "__main__":
    main()

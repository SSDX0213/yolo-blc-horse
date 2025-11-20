import os
from ultralytics import YOLO

# 遍历 ./Result 目录下所有文件
root_dir = "/root/ultralytics-8.3.27/horse_models_lite"

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "best.pt":
            model_path = os.path.join(root, file)
            print(f"\n📁 模型路径: {model_path}")
            try:
                model = YOLO(model_path)
                metrics = model.val()
                print(f"  Precision: {metrics.box.map50:.4f}")
                print(f"  Recall:    {metrics.box.map:.4f}")
                print(f"  mAP@50:    {metrics.box.map50:.4f}")
                print(f"  mAP@50-95: {metrics.box.map:.4f}")
            except Exception as e:
                print(f"  ❌ 加载模型出错: {e}")
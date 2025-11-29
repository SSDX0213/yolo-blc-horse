import os
from ultralytics import YOLO

# =============================================================
# 配置：当前目录下的图片文件名
# =============================================================
IMAGE_FILE = "18Z073正视图.jpg"   # ← 你要检测的那一张图

# 你的三个模型路径
MODELS = {
    "part": "model/part/orin.pt",
    "head": "model/head/orin.pt",
    "head_feature": "model/head_feature/orin.pt"
}

OUTPUT_DIR = "vis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 对单张图运行三个模型，分别保存结果
# =============================================================
def run_inference():
    if not os.path.exists(IMAGE_FILE):
        print(f"❌ 错误：当前目录下找不到 {IMAGE_FILE}")
        return

    for model_name, model_path in MODELS.items():
        print(f"\n========== 正在运行模型：{model_name} ==========")
        model = YOLO(model_path)

        save_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_dir, exist_ok=True)

        model.predict(
            IMAGE_FILE,
            save=True,
            project=save_dir,
            name="",         # 直接将输出保存到文件夹内
            exist_ok=True,
            conf=0.05        # 降低阈值，尽量保证检出
        )

    print("\n🎉 推理完成！所有结果已保存在：")
    print(f"📁 {OUTPUT_DIR}")


if __name__ == "__main__":
    run_inference()

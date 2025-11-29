import cv2
from ultralytics import YOLO

# 1. 加载模型
# 确保 'yolov11n-seg.pt' 文件在您运行此脚本的目录下
model = YOLO('./horse_models_lite/orin_v11n/weights/best.pt')

# 2. 指定图片路径
# 替换 'your_image.jpg' 为您要分割的图片文件名
image_path = 'test.jpg' 

# 3. 进行预测和分割
# source: 指定输入，conf: 置信度阈值，save: 自动保存结果图片
results = model.predict(
    source=image_path, 
    conf=0.25, 
    save=True
)

# 4. 结果说明
# 结果图片会保存在一个名为 'runs/segment/predict' 的文件夹中。
print("✅ 分割完成！")
print("结果图片已自动保存到 'runs/segment/predict' 目录下。")

# 可选：打印检测到的对象信息
for r in results:
    print(f"检测到 {len(r.boxes)} 个对象。")
    if len(r.masks):
        print(f"检测到 {len(r.masks)} 个分割掩码。")
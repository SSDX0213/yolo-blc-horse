import os
import shutil

# 根目录
BASE = "/root/ultralytics-8.0"
ALLPHOTOS = os.path.join(BASE, "allphotos")
PART = os.path.join(BASE, "photoslabels", "horse")
LABELS_TRAIN = os.path.join(PART, "labels", "train")
LABELS_VAL = os.path.join(PART, "labels", "val")
IMAGES_TRAIN = os.path.join(PART, "images", "train")
IMAGES_VAL = os.path.join(PART, "images", "val")

# 支持的图片扩展名，你可以根据你实际情况增加
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def copy_images(label_dir, target_img_dir):
    ensure_dir(target_img_dir)
    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        # 在 allphotos 中查找同名图片
        found = False
        for ext in IMG_EXTS:
            imgname = base + ext
            src = os.path.join(ALLPHOTOS, imgname)
            if os.path.exists(src):
                dst = os.path.join(target_img_dir, imgname)
                # 方法一：copy
                # shutil.copy2(src, dst)
                shutil.move(src, dst)
                found = True
                break
        if not found:
            print(f"警告：未找到图片对应于标签 {fname}")

def main():
    print("拷贝 train 图片 …")
    copy_images(LABELS_TRAIN, IMAGES_TRAIN)
    print("拷贝 val 图片 …")
    copy_images(LABELS_VAL, IMAGES_VAL)
    print("完成。")

if __name__ == "__main__":
    main()

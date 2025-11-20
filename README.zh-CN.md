# OK
这里是中文文档

# 训练
运行同目录下train_v8即可

## 每次训练前需要修改什么
1. 数据配置文件默认用horse.yaml，无特殊情况不用改，有的话我会跟你说
2. name每次修改成需要跑的模型的name
3. 加载模型里面的模型路径，根据要跑的模型切换，当前我在项目根目录下塞了5个yaml，都是yolo-seg打头，你用什么yaml，name就换成对应的，方便我们后续记录结果，Orin即指代原生的
4. 当前服务器上已经跑完了Ghost，我这里已经跑完了PCC，Orin，PCC-LRASPP，剩下的就是LRASPP

## 所以复制以下的内容替换train_v8即可


### 第二批
```python
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = YOLO("./yolov8-seg-LRASPP.yaml")

    # 3. 训练（默认激活函数）
    model.train(
        data="./horse.yaml",  # 你的数据配置文件
        epochs=200,
        imgsz=640,
        batch=64,
        project="horse_models_lite",
        name="LRASPP",
        device="0",
        lr0=0.002,
        lrf=0.01,
        cache="ram"
    )
```

## 数据集路径

你打开horse.yaml，把上面那个路径改到给你的数据集下的labelphotos/horse即可，记得前面还有一个labelsphotos

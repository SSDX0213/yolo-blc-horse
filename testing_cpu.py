import os
import time
import torch
import pandas as pd
from ultralytics import YOLO
from thop import profile


# ============================================
#  统计模型参数量 + FLOPs（CPU）
# ============================================
def get_model_complexity(model):
    model = model.cpu()   # 强制放 CPU
    params = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        dummy = torch.zeros(1, 3, 640, 640)  # CPU 输入
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = flops / 1e9  # GFLOPs
    except Exception:
        flops = None

    return params, flops


# ============================================
#  单张图片推理时间（毫秒）——CPU版本
# ============================================
def get_inference_time(model):
    model = model.cpu()       # 强制 CPU
    dummy = torch.zeros(1, 3, 640, 640)

    # 预热
    for _ in range(3):
        _ = model(dummy)

    # 开始计时
    t0 = time.time()
    _ = model(dummy)
    t1 = time.time()

    infer_time_ms = (t1 - t0) * 1000
    return infer_time_ms


# ============================================
#  自动验证模型（只算检测指标，不用GPU）
# ============================================
def evaluate_model(yolo_model_path):
    try:
        model = YOLO(yolo_model_path)
        model.to("cpu")  # 强制 CPU 验证（更慢但更稳定）
        
        metrics = model.val(device="cpu")  # 这里也明确指定 CPU
        return (
            float(metrics.box.mp),      # Precision
            float(metrics.box.mr),      # Recall
            float(metrics.box.f1),      # F1
            float(metrics.box.map50),   # mAP50
            float(metrics.box.map),     # mAP50-95
            model
        )
    except Exception:
        return None, None, None, None, None, None


# ============================================
#  主函数：遍历模型文件夹
# ============================================
def main():
    ROOT = "horse_models_lite_all"
    results = []

    print("\n====== 多模型验证开始（CPU推理） ======\n")

    for root, dirs, files in os.walk(ROOT):
        for f in files:
            if f.endswith("best.pt"):
                path = os.path.join(root, f)
                print(f"📁 模型路径: {path}")

                # 执行验证
                P, R, F1, m50, m50_95, loaded_model = evaluate_model(path)

                if loaded_model is not None:
                    # Params + FLOPs
                    params, flops = get_model_complexity(loaded_model.model)

                    # 单张图片推理时间（CPU）
                    infer_time = get_inference_time(loaded_model.model)

                    print(f"  ✔️ 完成验证 | Params={params:.3f}M  FLOPs={flops:.3f}G  Infer(CPU)={infer_time:.2f}ms")
                else:
                    print("  ❌ 模型验证失败（跳过参数/FLOPs）")
                    params = flops = infer_time = None

                results.append([
                    path,
                    root.split("/")[-1],
                    P, R, F1, m50, m50_95,
                    params, flops, infer_time
                ])
    
    # 保存结果表
    df = pd.DataFrame(results, columns=[
        "Model Path", "Type",
        "Precision", "Recall", "F1",
        "mAP50", "mAP50-95",
        "Params(M)", "FLOPs(G)",
        "Infer Time (ms, CPU)"
    ])

    print("\n====== 测试结束，生成汇总表（CPU推理） ======\n")
    print(df)

    df.to_csv("model_summary_cpu.csv", index=False)
    df.to_excel("model_summary_cpu.xlsx", index=False)

    print("📄 CPU 版本汇总 CSV 已保存！")
    print("📘 CPU 版本汇总 Excel 已保存！\n")
    print("🎉 全部完成（CPU 推理版本）！")


if __name__ == "__main__":
    main()

import os
import time
import torch
import pandas as pd
from ultralytics import YOLO
from thop import profile


# ============================================
#  获取模型参数量 + FLOPs
# ============================================
def get_model_complexity(model):
    params = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        dummy = torch.zeros(1, 3, 640, 640).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = flops / 1e9  # GFLOPs
    except Exception:
        flops = None

    return params, flops


# ============================================
#  推理时间
# ============================================
def get_inference_time(model):
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, 640, 640).to(device)

    for _ in range(3):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) * 1000


# ============================================
#  评估 SEG 模型指标（含每类 F1）
# ============================================
def evaluate_model(model_path):
    model = YOLO(model_path)
    metrics = model.val()

    if metrics.seg:
        P = float(metrics.seg.mp)
        R = float(metrics.seg.mr)
        m50 = float(metrics.seg.map50)
        m50_95 = float(metrics.seg.map)

        # F1 是 ndarray，保留全部类别
        F1_array = metrics.seg.f1
        F1_mean = float(F1_array.mean())

        return P, R, F1_mean, m50, m50_95, F1_array, model

    return None, None, None, None, None, None, model


# ============================================
#  主程序
# ============================================
def main():
    ROOT = "horse_models_lite_all"
    results = []

    print("\n====== 多模型验证开始 ======\n")

    for root, dirs, files in os.walk(ROOT):
        for f in files:
            if f.endswith("best.pt"):
                path = os.path.join(root, f)

                # 模型名称 —— 只保留最后一级目录
                model_name = os.path.basename(os.path.dirname(os.path.dirname(path)))

                print(f"📁 模型: {model_name}")

                # 模型验证
                P, R, F1_mean, m50, m50_95, F1_array, loaded_model = evaluate_model(path)

                # 参数量 + FLOPs
                params, flops = get_model_complexity(loaded_model.model)

                # 速度
                infer_time = get_inference_time(loaded_model.model)

                # 展开每个 F1
                f1_dict = {}
                if F1_array is not None:
                    for i, v in enumerate(F1_array):
                        f1_dict[f"F1_cls{i}"] = float(v)

                # 汇总一行
                entry = {
                    "Model": model_name,
                    "Precision": P,
                    "Recall": R,
                    "F1_mean": F1_mean,
                    "mAP50": m50,
                    "mAP50-95": m50_95,
                    "Params(M)": params,
                    "FLOPs(G)": flops,
                    "Infer Time(ms)": infer_time,
                }
                entry.update(f1_dict)  # 加入每类 F1

                results.append(entry)

    # 保存表格
    df = pd.DataFrame(results)

    print("\n====== 结果表 ======\n")
    print(df)

    df.to_csv("model_summary.csv", index=False)
    df.to_excel("model_summary.xlsx", index=False)

    print("📄 CSV 已保存！")
    print("📘 Excel 已保存！")
    print("🎉 完成！")


if __name__ == "__main__":
    main()

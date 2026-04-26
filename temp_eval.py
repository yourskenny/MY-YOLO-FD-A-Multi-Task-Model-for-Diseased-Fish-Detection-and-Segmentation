import pandas as pd
import glob
import os

runs_dir = r"C:\coding\YOLO-FD\YOLO-FD-main (1)(1)\YOLO-FD-main\runs\detect"
train_dirs = sorted(glob.glob(os.path.join(runs_dir, "train*")), key=os.path.getmtime, reverse=True)[:8]

print("📊 最近几轮自动化训练跑分汇总：")
print("-" * 50)
for d in train_dirs:
    csv_path = os.path.join(d, "results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            best_mAP50 = df.iloc[:, 6].max()
            best_mAP50_95 = df.iloc[:, 7].max()
            epoch = df.iloc[:, 6].idxmax()
            print(f"📁 {os.path.basename(d).ljust(10)} | mAP50: {best_mAP50:.5f} (Epoch {epoch}) | mAP50-95: {best_mAP50_95:.5f}")
        except Exception as e:
            print(f"📁 {os.path.basename(d).ljust(10)} | [未完成或解析失败]")
print("-" * 50)

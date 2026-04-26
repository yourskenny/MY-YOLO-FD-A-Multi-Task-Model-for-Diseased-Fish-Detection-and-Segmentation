import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

runs_dir = "runs/detect"
train_dirs = ['train28', 'train29', 'train30']

plt.figure(figsize=(15, 10))

for i, tdir in enumerate(train_dirs):
    csv_path = os.path.join(runs_dir, tdir, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 清理列名空格
        df.columns = df.columns.str.strip()
        
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train/box_loss'], label=f'{tdir} train')
        plt.plot(df['epoch'], df['val/box_loss'], linestyle='--', label=f'{tdir} val')
        plt.title('Box Loss Trend')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['train/cls_loss'], label=f'{tdir} train')
        plt.plot(df['epoch'], df['val/cls_loss'], linestyle='--', label=f'{tdir} val')
        plt.title('Class Loss Trend')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label=f'{tdir} mAP50(B)')
        if 'metrics/mAP50(M)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(M)'], linestyle='--', label=f'{tdir} mAP50(M)')
        plt.title('mAP50 Trend')
        plt.legend()

plt.tight_layout()
plt.savefig("new_arch_trends.png")
print("Saved trend plots to new_arch_trends.png")

# 检查最后一轮的详细指标
last_csv = os.path.join(runs_dir, 'train30', "results.csv")
if os.path.exists(last_csv):
    df = pd.read_csv(last_csv)
    df.columns = df.columns.str.strip()
    print("\nTrain30 后期（最后5轮）的指标：")
    print(df[['epoch', 'train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].tail(5))

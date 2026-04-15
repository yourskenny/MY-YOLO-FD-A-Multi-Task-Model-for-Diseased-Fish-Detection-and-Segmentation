import os
import sys
import ctypes
import pandas as pd
import glob
import time

# 确保能找到 ultralytics 包
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ultralytics.yolo.v8.detect import DetectionTrainer

def prevent_sleep():
    """防止 Windows 进入睡眠模式"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[INFO] Anti-sleep mode enabled.")
    except Exception as e:
        pass

def get_latest_run():
    """获取最新的一轮训练输出目录"""
    runs_dir = os.path.join(project_dir, 'runs', 'detect')
    if not os.path.exists(runs_dir):
        return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs:
        return None
    latest_dir = max(train_dirs, key=os.path.getmtime)
    return latest_dir

def evaluate_latest_run(latest_dir):
    """读取 results.csv 提取 mAP50 和 mAP50-95 的最佳跑分"""
    if not latest_dir:
        return 0.0, 0.0
    results_file = os.path.join(latest_dir, 'results.csv')
    if not os.path.exists(results_file):
        return 0.0, 0.0
    
    try:
        df = pd.read_csv(results_file)
        # 清理列名空格
        df.columns = df.columns.str.strip()
        
        map50_col = [c for c in df.columns if 'mAP50(B)' in c][0]
        map50_95_col = [c for c in df.columns if 'mAP50-95(B)' in c][0]
        
        best_map50 = df[map50_col].max()
        best_map50_95 = df[map50_95_col].max()
        return float(best_map50), float(best_map50_95)
    except Exception as e:
        print(f"[ERROR] Could not parse results: {e}")
        return 0.0, 0.0

def run_auto_tuner():
    prevent_sleep()
    
    target_map50 = 0.96
    target_map50_95 = 0.80
    
    # 定义阶梯式自动调参策略 (Progressive Tuning Strategies)
    strategies = [
        # Round 1: 移除破坏性增强(MixUp/CopyPaste)，提升图像分辨率至 800，关闭可能影响收敛的 PCGrad
        {
            "imgsz": 800, "mixup": 0.0, "copy_paste": 0.0, "pcgrad": False, 
            "lr0": 0.01, "pre_weights": "yolo-fd.pt", "epochs": 150
        },
        # Round 2: 如果 800 没达标，直接用 5070Ti 的大显存暴力拉升到 1024 极致分辨率
        {
            "imgsz": 1024, "mixup": 0.0, "copy_paste": 0.0, "pcgrad": False, 
            "lr0": 0.01, "pre_weights": "yolo-fd.pt", "epochs": 150
        },
        # Round 3: 如果还没达标，退回 800 但开启 PCGrad (多任务梯度投影) 尝试平滑损失
        {
            "imgsz": 800, "mixup": 0.0, "copy_paste": 0.0, "pcgrad": True, 
            "lr0": 0.01, "pre_weights": "yolo-fd.pt", "epochs": 150
        },
        # Round 4: 利用更小的学习率 (lr=0.001) 从最好的权重微调，加入轻微的 mixup=0.1
        {
            "imgsz": 800, "mixup": 0.1, "copy_paste": 0.0, "pcgrad": False, 
            "lr0": 0.001, "pre_weights": "yolo-fd.pt", "epochs": 150
        }
    ]
    
    print(f"===========================================================")
    print(f"🤖 YOLO-FD Auto-Tuning Agent 启动")
    print(f"🎯 目标: mAP50 >= {target_map50}, mAP50-95 >= {target_map50_95}")
    print(f"===========================================================\n")
    
    for i, config in enumerate(strategies):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"🚀 正在启动第 {i+1}/{len(strategies)} 轮自动化训练")
        print(f"⚙️ 当前采用策略参数: {config}")
        
        args = dict(
            model='yolov8-gold.yaml',
            data='norcardis_disease.yaml',
            device='0',
            batch=16,
            mtl=1,
            cagrad=False,
            deterministic=False,
            resume=False,
            warmup_epochs=3.0,
            **config
        )
        
        try:
            trainer = DetectionTrainer(overrides=args)
            trainer.train()
        except Exception as e:
            print(f"[ERROR] Training failed for round {i+1}: {e}")
            continue
            
        # 评估本轮成绩
        latest_dir = get_latest_run()
        best_map50, best_map50_95 = evaluate_latest_run(latest_dir)
        
        print(f"\n📊 第 {i+1} 轮最终跑分: mAP50 = {best_map50:.5f}, mAP50-95 = {best_map50_95:.5f}")
        
        if best_map50 >= target_map50 and best_map50_95 >= target_map50_95:
            print(f"🎉 太棒了！目标跑分已在第 {i+1} 轮达成！")
            print(f"📦 达标模型权重保存在: {latest_dir}\\weights\\best.pt")
            break
        else:
            print(f"⚠️ 成绩未达标 (距离目标 mAP50={target_map50}, mAP50-95={target_map50_95} 还有差距)。")
            if i < len(strategies) - 1:
                print("🔄 准备调整参数，将在 10 秒后自动启动下一轮方案...")
                time.sleep(10)
            else:
                print("🏁 所有预设方案均已跑完，但未能完全达标，请分析日志后添加新策略。")

if __name__ == '__main__':
    run_auto_tuner()
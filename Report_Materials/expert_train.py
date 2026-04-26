import os
import multiprocessing
import cv2

# Windows 兼容与死锁保护
multiprocessing.freeze_support()
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ['PIN_MEMORY'] = 'False'

from ultralytics.yolo.v8.detect import DetectionTrainer

def run_expert_training():
    print("🧠 正在启动【小目标专家级】修正版训练管线...")
    print("🛡️ 核心策略 1: 发现网络魔改极大(仅命中53层)，彻底解冻 Backbone 并调高 lr0，让新结构能顺利起飞！")
    print("🛡️ 核心策略 2: DFL 分布焦点损失强制提拉，专门针对 mAP50-95 的像素级回归痛点。")
    print("🛡️ 核心策略 3: 削弱 Mixup 防止细微病灶在重叠时被过度抹除，保障原始图像质量。")
    
    args = dict(
        model='yolov8-gold.yaml',
        data='norcardis_disease.yaml',
        device='0',
        batch=8,
        imgsz=800,
        workers=4,
        mtl=1,
        cagrad=False,
        deterministic=False, # 关闭确定性训练，因为部分CUDA算子不支持
        resume=False,
        pretrained=True,
        pre_weights='yolo-fd.pt',
        epochs=150,
        amp=False,
        
        # --- 修正版专家级超参数 ---
        # 1. 解冻 Backbone 与提升学习率
        freeze=0,            # 【修正】因为 yolov8-gold.yaml 相比 yolo-fd.pt 改动太大(只加载了53层)，必须全量微调！
        lr0=0.005,           # 【修正】提升初始学习率，让全新初始化的 Neck 和 P2 Head 能够快速收敛
        lrf=0.01,            # 最终学习率衰减到 1%
        warmup_epochs=3.0,   # 恢复正常预热时间
        
        # 2. 降噪与聚焦小目标
        box=7.5,             # 维持较高的回归框权重
        dfl=2.0,             # 强制拉高 DFL 分布焦点损失，追求像素级精准定位 (提升 mAP50-95)
        cls=0.5,             # 降低普通分类权重，防止背景框喧宾夺主
        
        # 3. 适度的数据增强 (防止过度破坏小目标)
        mosaic=1.0,          # 100% 概率将 4 张图拼在一起
        mixup=0.05,          # 【修正】降低 MixUp，防止小病灶被背景融合彻底掩盖
        copy_paste=0.05,     # 【修正】降低 Copy-Paste 概率
        
        # 其他优化
        weight_decay=0.0005, # 防止过拟合
        label_smoothing=0.05 # 降低标签平滑，保持对小目标的置信度
    )
    
    try:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        print("🎉 专家级训练管线已顺利完成！请查看 runs/detect 下的最新跑分。")
    except Exception as e:
        print(f"[ERROR] 专家级训练崩溃: {e}")

if __name__ == "__main__":
    run_expert_training()

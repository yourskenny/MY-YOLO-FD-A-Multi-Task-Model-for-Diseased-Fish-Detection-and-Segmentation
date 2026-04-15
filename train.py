import ctypes
import time
from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils import DEFAULT_CFG


def prevent_sleep():
    """Prevent Windows from going to sleep during training."""
    try:
        # ES_CONTINUOUS = 0x80000000, ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[INFO] Anti-sleep mode enabled: The system will stay awake during training.")
    except Exception as e:
        print("[WARNING] Could not enable anti-sleep mode:", e)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    prevent_sleep()
    model = cfg.model
    model = 'yolov8-gold.yaml'
    data =  'norcardis_disease.yaml'
    device = '0'
    freeze = 0
    epochs = 150
    pre_weights = r'runs\detect\train12\weights\best.pt'  # 从上一轮最好的权重开始微调
    batch = 16
    mtl = 1  # 0 equal, 1 uncertainty
    pcgrad = True
    cagrad = False
    deterministic = False
    resume = False  # 这里是新一轮微调，不是断点续训，设为 False
    
    # 新增数据增强与损失超参数调优
    mixup = 0.15
    copy_paste = 0.1
    warmup_epochs = 5.0
    weight_decay = 0.001
    lr0 = 0.001  # 微调时使用较小的初始学习率
    lrf = 0.01
    
    # Enable resume if a specific checkpoint is provided instead of the base pre_weights
    if resume and pre_weights.endswith('last.pt'):
        model = pre_weights
    
    args = dict(model=model,
                data=data,
                device=device,
                pre_weights=pre_weights,
                freeze=freeze,
                epochs=epochs,
                batch=batch,
                mtl=mtl,
                pcgrad=pcgrad,
                cagrad=cagrad,
                deterministic=deterministic,
                resume=resume,
                mixup=mixup,
                copy_paste=copy_paste,
                warmup_epochs=warmup_epochs,
                weight_decay=weight_decay,
                lr0=lr0,
                lrf=lrf)
    
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            if use_python:
                from ultralytics import YOLO
                YOLO(model).train(**args)
            else:
                trainer = DetectionTrainer(overrides=args)
                trainer.train()
            
            print("[INFO] Training completed successfully!")
            break  # Break out of the retry loop if successful
            
        except Exception as e:
            print(f"[ERROR] Training interrupted (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Ensure resume is turned on for subsequent attempts
                args['resume'] = True
                args['model'] = r'runs\detect\train12\weights\last.pt'
                args['pre_weights'] = r'runs\detect\train12\weights\last.pt'
            else:
                print("[ERROR] Max retries reached. Training failed.")
                raise e


if __name__ == '__main__':
    train()
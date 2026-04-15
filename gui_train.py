import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import sys
import threading
import ctypes

class YOLOTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO-FD 多任务模型训练控制台")
        self.root.geometry("700x800")
        self.root.resizable(False, False)
        
        self.process = None
        self.training_thread = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # --- 基本配置区域 ---
        frame_basic = ttk.LabelFrame(self.root, text="基本配置 (Basic Config)", padding=10)
        frame_basic.pack(fill=tk.X, padx=10, pady=5)
        
        # Model
        ttk.Label(frame_basic, text="模型结构 (model):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_model = tk.StringVar(value='yolov8-gold.yaml')
        ttk.Entry(frame_basic, textvariable=self.var_model, width=30).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Data
        ttk.Label(frame_basic, text="数据集 (data):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_data = tk.StringVar(value='norcardis_disease.yaml')
        ttk.Entry(frame_basic, textvariable=self.var_data, width=30).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Pre-weights
        ttk.Label(frame_basic, text="预训练权重 (pre_weights):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_weights = tk.StringVar(value=r'runs\detect\train12\weights\best.pt')
        ttk.Entry(frame_basic, textvariable=self.var_weights, width=40).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Button(frame_basic, text="浏览...", command=self.browse_weights).grid(row=2, column=2, padx=5)
        
        # Device
        ttk.Label(frame_basic, text="设备 (device):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_device = tk.StringVar(value='0')
        ttk.Entry(frame_basic, textvariable=self.var_device, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # --- 超参数区域 ---
        frame_hyper = ttk.LabelFrame(self.root, text="超参数 (Hyperparameters)", padding=10)
        frame_hyper.pack(fill=tk.X, padx=10, pady=5)
        
        # Epochs & Batch
        ttk.Label(frame_hyper, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_epochs = tk.IntVar(value=150)
        ttk.Entry(frame_hyper, textvariable=self.var_epochs, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(frame_hyper, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=10)
        self.var_batch = tk.IntVar(value=16)
        ttk.Entry(frame_hyper, textvariable=self.var_batch, width=10).grid(row=0, column=3, sticky=tk.W, pady=2)
        
        # LR
        ttk.Label(frame_hyper, text="初始学习率 (lr0):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_lr0 = tk.DoubleVar(value=0.001)
        ttk.Entry(frame_hyper, textvariable=self.var_lr0, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(frame_hyper, text="最终学习率 (lrf):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=10)
        self.var_lrf = tk.DoubleVar(value=0.01)
        ttk.Entry(frame_hyper, textvariable=self.var_lrf, width=10).grid(row=1, column=3, sticky=tk.W, pady=2)
        
        # Augmentation
        ttk.Label(frame_hyper, text="MixUp:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_mixup = tk.DoubleVar(value=0.15)
        ttk.Entry(frame_hyper, textvariable=self.var_mixup, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(frame_hyper, text="Copy-Paste:").grid(row=2, column=2, sticky=tk.W, pady=2, padx=10)
        self.var_copypaste = tk.DoubleVar(value=0.1)
        ttk.Entry(frame_hyper, textvariable=self.var_copypaste, width=10).grid(row=2, column=3, sticky=tk.W, pady=2)
        
        # --- 多任务与高级设置 ---
        frame_adv = ttk.LabelFrame(self.root, text="高级设置 (Advanced)", padding=10)
        frame_adv.pack(fill=tk.X, padx=10, pady=5)
        
        self.var_mtl = tk.IntVar(value=1)
        ttk.Checkbutton(frame_adv, text="启用不确定性权重 (MTL=1)", variable=self.var_mtl).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.var_pcgrad = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_adv, text="启用 PCGrad", variable=self.var_pcgrad).grid(row=0, column=1, sticky=tk.W, pady=2, padx=20)
        
        self.var_resume = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_adv, text="断点续训 (Resume)", variable=self.var_resume).grid(row=0, column=2, sticky=tk.W, pady=2, padx=20)
        
        # --- 控制按钮区域 ---
        frame_ctrl = ttk.Frame(self.root, padding=10)
        frame_ctrl.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_start = ttk.Button(frame_ctrl, text="▶ 开始训练", command=self.start_training, width=20)
        self.btn_start.pack(side=tk.LEFT, padx=20)
        
        self.btn_stop = ttk.Button(frame_ctrl, text="⏹ 停止训练", command=self.stop_training, state=tk.DISABLED, width=20)
        self.btn_stop.pack(side=tk.RIGHT, padx=20)
        
        # --- 日志输出区域 ---
        frame_log = ttk.LabelFrame(self.root, text="训练日志 (Logs)", padding=10)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_log = tk.Text(frame_log, wrap=tk.WORD, height=15, bg="black", fg="white", font=("Consolas", 10))
        self.text_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(frame_log, command=self.text_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_log.config(yscrollcommand=scrollbar.set)
        
    def browse_weights(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="选择预训练权重", filetypes=(("PyTorch Models", "*.pt"), ("All Files", "*.*")))
        if filename:
            # 尽量使用相对路径
            rel_path = os.path.relpath(filename, os.getcwd())
            self.var_weights.set(rel_path)
            
    def log(self, message):
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.root.update_idletasks()
        
    def generate_train_script(self):
        """生成一个临时的训练执行脚本以独立进程运行"""
        project_dir = os.path.abspath(os.path.dirname(__file__))
        script_content = f'''import ctypes
import time
import sys
import os

# 将当前项目目录添加到系统路径中，确保能找到本地的 ultralytics 包
sys.path.insert(0, r"{project_dir}")

from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils import DEFAULT_CFG

def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[INFO] Anti-sleep mode enabled.")
    except Exception as e:
        print("[WARNING] Could not enable anti-sleep mode:", e)

def run_train():
    prevent_sleep()
    
    args = dict(
        model='{self.var_model.get()}',
        data='{self.var_data.get()}',
        device='{self.var_device.get()}',
        pre_weights=r'{self.var_weights.get()}',
        freeze=0,
        epochs={self.var_epochs.get()},
        batch={self.var_batch.get()},
        mtl={self.var_mtl.get()},
        pcgrad={self.var_pcgrad.get()},
        cagrad=False,
        deterministic=False,
        resume={self.var_resume.get()},
        mixup={self.var_mixup.get()},
        copy_paste={self.var_copypaste.get()},
        warmup_epochs=5.0,
        weight_decay=0.001,
        lr0={self.var_lr0.get()},
        lrf={self.var_lrf.get()}
    )
    
    max_retries = 3
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            trainer = DetectionTrainer(overrides=args)
            trainer.train()
            print("[INFO] Training completed successfully!")
            break
        except Exception as e:
            print(f"[ERROR] Training interrupted (Attempt {{attempt + 1}}/{{max_retries}}): {{e}}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {{retry_delay}} seconds...")
                time.sleep(retry_delay)
                args['resume'] = True
            else:
                print("[ERROR] Max retries reached. Training failed.")
                sys.exit(1)

if __name__ == '__main__':
    run_train()
'''
        with open("gui_train_runner.py", "w", encoding="utf-8") as f:
            f.write(script_content)
            
    def start_training(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("警告", "训练已经在运行中！")
            return
            
        self.generate_train_script()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.text_log.delete(1.0, tk.END)
        self.log("🚀 开始准备训练环境...")
        self.log("读取配置并启动后台进程...")
        
        # 防止睡眠 (GUI端也加一份)
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except:
            pass
            
        self.training_thread = threading.Thread(target=self._run_process)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def _run_process(self):
        # 启动子进程
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        # 设置 PYTHONPATH 环境变量，避免依赖找不到
        env = os.environ.copy()
        project_dir = os.path.abspath(os.path.dirname(__file__))
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_dir
            
        self.process = subprocess.Popen(
            [sys.executable, "gui_train_runner.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            startupinfo=startupinfo,
            env=env,
            cwd=project_dir,  # 确保在当前目录执行
            encoding='utf-8',
            errors='replace'
        )
        
        # 实时读取输出
        for line in self.process.stdout:
            self.root.after(0, self.log, line.strip())
            
        self.process.wait()
        
        self.root.after(0, self._training_finished)
        
    def _training_finished(self):
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.process.returncode == 0:
            self.log("✅ 训练流程已结束！")
            messagebox.showinfo("完成", "模型训练已经完成！")
        else:
            self.log("❌ 训练被终止或出现异常。")
            
    def stop_training(self):
        if self.process is not None and self.process.poll() is None:
            if messagebox.askyesno("确认停止", "确定要强制终止当前正在进行的训练吗？\n(这可能会导致当前 epoch 的进度丢失)"):
                # Windows 下强制杀死进程树
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                self.log("⚠️ 已发送强制终止指令！")
                self.btn_start.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTrainerGUI(root)
    root.mainloop()
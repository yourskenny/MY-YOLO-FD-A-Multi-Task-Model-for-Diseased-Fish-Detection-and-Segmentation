import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys

PROJECT_DIR = r"C:\coding\YOLO-FD\YOLO-FD-main (1)(1)\YOLO-FD-main"

class AgentDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 YOLO-FD Agent 监控中心")
        self.root.geometry("500x320")
        
        # 居中显示
        self.root.eval('tk::PlaceWindow . center')
        
        # 标题
        tk.Label(root, text="YOLO-FD 大模型自动调参控制台", font=("微软雅黑", 16, "bold"), fg="#333").pack(pady=20)
        
        # 按钮区样式
        style = ttk.Style()
        style.configure("TButton", font=("微软雅黑", 11), padding=6)
        
        # 按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=40)

        ttk.Button(btn_frame, text="▶️ 启动 Agent 训练 (弹出独立可见黑框终端)", command=self.start_agent).pack(fill=tk.X, pady=8)
        ttk.Button(btn_frame, text="📊 启动 TensorBoard 可视化训练曲线", command=self.start_tensorboard).pack(fill=tk.X, pady=8)
        ttk.Button(btn_frame, text="⏹ 一键强制停止所有训练", command=self.stop_all).pack(fill=tk.X, pady=8)

        # 底部状态
        self.status_label = tk.Label(root, text="状态: 待机中", font=("微软雅黑", 10), fg="gray")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def start_agent(self):
        self.status_label.config(text="状态: 正在清理旧进程并启动独立监控窗口...", fg="blue")
        self.root.update()
        
        # 杀掉可能隐藏在后台的进程
        os.system('powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match \'llm_auto_tuner.py\' } | Stop-Process -Force -ErrorAction SilentlyContinue"')
        
        # 使用 sys.executable 确保弹出的控制台使用的是和当前完全一致的 Python 虚拟环境，避免找不到依赖包
        python_exe = sys.executable
        subprocess.Popen(f'start "YOLO-FD Agent 实时监控" cmd /k "{python_exe} llm_auto_tuner.py"', shell=True, cwd=PROJECT_DIR)
        
        messagebox.showinfo("启动成功", "已为您弹出专属的控制台窗口！\n\n您可以在弹出的黑框中实时查看 PyTorch 的进度条和 Agent 的分析日志。")
        self.status_label.config(text="状态: Agent 训练正在独立窗口中运行", fg="green")

    def start_tensorboard(self):
        self.status_label.config(text="状态: 正在启动 TensorBoard 服务...", fg="blue")
        self.root.update()
        
        # 同样使用 sys.executable 确保 TensorBoard 使用当前环境
        python_exe = sys.executable
        subprocess.Popen(f'start "TensorBoard 服务端" cmd /k "{python_exe} -m tensorboard.main --logdir=runs/detect"', shell=True, cwd=PROJECT_DIR)
        
        # 延迟 3 秒后自动在浏览器中打开网页
        import webbrowser
        import time
        import threading
        
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:6006")
            self.root.after(0, lambda: self.status_label.config(text="状态: TensorBoard 浏览器已打开", fg="green"))
            
        threading.Thread(target=open_browser, daemon=True).start()

    def stop_all(self):
        self.status_label.config(text="状态: 正在终止所有相关进程...", fg="red")
        self.root.update()
        
        os.system('powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match \'llm_auto_tuner.py\' } | Stop-Process -Force -ErrorAction SilentlyContinue"')
        os.system('powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match \'tensorboard\' } | Stop-Process -Force -ErrorAction SilentlyContinue"')
        
        messagebox.showinfo("已停止", "所有的 Agent 训练进程与 TensorBoard 进程已被强制停止。")
        self.status_label.config(text="状态: 所有进程已安全停止", fg="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentDashboard(root)
    root.mainloop()
import os
import sys
import ctypes
import pandas as pd
import glob
import time
import json
import requests

# 确保能找到 ultralytics 包
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ultralytics.yolo.v8.detect import DetectionTrainer

# ==========================================
# LLM Agent 配置区 (Volcengine Ark API)
# ==========================================
API_KEY = "83ea63aa-c768-44e6-b73d-fa125b7dd49e"
API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
MODEL_NAME = "doubao-seed-2-0-pro-260215"

def prevent_sleep():
    """防止 Windows 进入睡眠模式或因息屏/合盖导致 GPU 挂起"""
    try:
        # ES_CONTINUOUS = 0x80000000 (保持状态直到明确清除)
        # ES_SYSTEM_REQUIRED = 0x00000001 (强制系统不进入睡眠)
        # ES_DISPLAY_REQUIRED = 0x00000002 (强制显示器不关闭，对合盖挂起有奇效)
        # ES_AWAYMODE_REQUIRED = 0x00000040 (离开模式，允许关闭屏幕但系统和设备全速运行)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002 | 0x00000040)
        print("[INFO] Anti-sleep mode (Away Mode + Display Required) enabled. You can close the lid safely.")
    except Exception as e:
        print(f"[WARNING] Failed to set anti-sleep mode: {e}")

def get_latest_run():
    runs_dir = os.path.join(project_dir, 'runs', 'detect')
    if not os.path.exists(runs_dir): return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs: return None
    return max(train_dirs, key=os.path.getmtime)

def get_last_interrupted_run():
    """获取最后一次中断的训练的 last.pt 路径"""
    runs_dir = os.path.join(project_dir, 'runs', 'detect')
    if not os.path.exists(runs_dir): return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs: return None
    latest_dir = max(train_dirs, key=os.path.getmtime)
    last_pt = os.path.join(latest_dir, 'weights', 'last.pt')
    if os.path.exists(last_pt):
        return last_pt
    return None

def evaluate_run(latest_dir):
    """读取结果并返回 mAP50, mAP50-95 以及最后几轮的 Loss 趋势以供大模型分析"""
    if not latest_dir: return 0.0, 0.0, ""
    results_file = os.path.join(latest_dir, 'results.csv')
    if not os.path.exists(results_file): return 0.0, 0.0, ""
    
    try:
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        map50_col = [c for c in df.columns if 'mAP50(B)' in c][0]
        map50_95_col = [c for c in df.columns if 'mAP50-95(B)' in c][0]
        
        best_map50 = float(df[map50_col].max())
        best_map50_95 = float(df[map50_95_col].max())
        
        # 截取最后5轮的关键数据给 LLM 分析
        tail_data = df[['epoch', 'train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss', map50_col, map50_95_col]].tail(5).to_string()
        return best_map50, best_map50_95, tail_data
    except Exception as e:
        print(f"[ERROR] 读取结果失败: {e}")
        return 0.0, 0.0, ""

def ask_llm_for_next_hyperparams(history_context, current_best_map50, current_best_map50_95, target_map50, target_map50_95):
    """
    调用 LLM API，让大模型分析历史数据并决定下一轮的超参数配置
    """
    prompt = f"""你是一个顶级的计算机视觉(YOLOv8)模型调参专家 Agent。
当前任务是在病鱼检测和分割数据集(小目标、细微病灶)上进行超参数搜索。
目标：mAP50 >= {target_map50}, mAP50-95 >= {target_map50_95}。
当前最佳：mAP50 = {current_best_map50}, mAP50-95 = {current_best_map50_95}。

【重要优化通知】：为了突破 0.936 的瓶颈，除了维持之前的架构升级（P2检测头+CBAM）外，我们刚刚在特征提取和损失函数层面做出了重大改进：
1. **替换为极轻量上下文聚合 (ContextAggregation)**：为了解决 P2 层极高分辨率带来的背景噪声（假阳性）问题以及训练极慢的瓶颈，我们已将 P2 和 P3 层替换为 **ContextAggregation**。它能极速聚合全局特征过滤噪声，让模型在满血多进程 (workers=4) 下高速训练。
2. **修复了 MTL（多任务学习）权重失衡**：之前训练后期检测任务的权重趋近于 0，现在已经通过 clamp 强制限制了 sigma 的发散，确保检测任务和分割任务均衡优化。
3. **引入了尺度自适应的 NWD 损失**：针对极小目标，NWD 对微小的位置偏移更加敏感，能够大幅提升小目标的召回率和高 IoU 阈值下的定位精度。

有了这三个层面的杀手锏，模型现在能够真正有效地学习极小病灶的特征且不受噪声干扰了！
请在分析时将这三个利好因素考虑进去，并给出一组**相对激进但又能防止过拟合**的新超参数，以便快速验证新机制的威力。

以下是历史实验和最近一轮的训练结果趋势：
{history_context}

请分析原因，并为下一轮训练提供一组新的超参数。
你必须严格返回合法的 JSON 格式数据，包含以下字段：
{{
    "analysis": "你的分析和理由(简短，请提及架构升级的预期影响)",
    "hyperparameters": {{
        "mixup": 0.1,  // 0.05 到 0.15 之间，用于增强小目标组合
        "copy_paste": 0.1, // 0.05 到 0.15 之间，专门应对小病灶稀缺
        "lr0": 0.01,   // 初始学习率
        "lrf": 0.01,   // 最终学习率系数
        "warmup_epochs": 3.0,
        "weight_decay": 0.0005,
        "pcgrad": false // 是否启用多任务梯度投影
    }}
}}
注意：不要输出任何 markdown 代码块标记，纯 JSON。"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        print("🧠 正在请求火山引擎 Doubao 大模型(Agent)进行分析决策...")
        # 增加超时限制 timeout=15 防止卡死
        response = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 适配 Codex Mirror API 的返回格式
        response_data = response.json()
        
        # 针对不同可能的响应结构进行兼容处理
        content = ""
        if 'output' in response_data:
            if isinstance(response_data['output'], dict):
                content = response_data['output'].get('text', '')
            elif isinstance(response_data['output'], list):
                for item in response_data['output']:
                    if item.get('type') == 'text':
                        content = item.get('text', '')
                        break
        elif 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice:
                if isinstance(choice['message'], dict):
                    content = choice['message'].get('content', '')
                elif isinstance(choice['message'], list):
                    for item in choice['message']:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            content = item.get('text', '')
                            break
            elif 'text' in choice:
                content = choice.get('text', '')
                
        if not content:
            print(f"❌ 无法从响应中解析内容: {response_data}")
            return {"mixup": 0.1, "copy_paste": 0.1, "pcgrad": False, "lr0": 0.005}
            
        content = content.strip()
        # 移除可能的 markdown 标记
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        
        result = json.loads(content)
        print(f"\n💡 Agent 分析: {result.get('analysis')}")
        return result.get('hyperparameters')
    except Exception as e:
        print(f"❌ Agent 请求失败: {e}")
        # 返回一个保守的默认备选参数
        return {"mixup": 0.1, "copy_paste": 0.1, "pcgrad": False, "lr0": 0.005}


def run_llm_agent():
    prevent_sleep()
    target_map50, target_map50_95 = 0.96, 0.80
    
    print(f"===========================================================")
    print(f"🤖 YOLO-FD 智能大模型炼丹 Agent (LLM-Driven) 已启动")
    print(f"🎯 终极目标: mAP50 >= {target_map50}, mAP50-95 >= {target_map50_95}")
    print(f"===========================================================\n")
    
    # 初始化历史上下文
    history_context = "第0轮 (基线): mAP50=0.933, mAP50-95=0.760。特征：细微的溃疡和结节病灶。\n"
    current_best_map50 = 0.933
    current_best_map50_95 = 0.760
    
    # 检查是否有上次中断的训练可以继续
    last_pt_path = get_last_interrupted_run()
    is_resuming_from_interruption = False
    
    # 强制不续训，因为我们修改了特征提取网络 (BiFormer) 和损失函数，让大模型重新适应
    if last_pt_path:
        print(f"🔄 检测到旧的权重: {last_pt_path}，但因 BiFormer 和 Loss 优化，本次强制重新开始训练！")

    for round_num in range(1, 11):  # 放开限制，准备进行正式的完整 150 轮多轮循环训练
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 准备第 {round_num} 轮训练...")
        
        if is_resuming_from_interruption and round_num == 1:
            print("▶️ 本轮将执行断点续训 (Resume)，跑完之前中断的 Epochs。")
            hyperparams = {}  # 续训时不需要新的超参数
            args = dict(
                model=last_pt_path,  # resume 时 model 和 pre_weights 都要指向 last.pt
                data='norcardis_disease.yaml',
                device='0',
                batch=8,
                workers=4,
                mtl=1,
                cagrad=False,
                deterministic=False,
                resume=True,  # 开启续训模式
                pre_weights=last_pt_path,
                epochs=150
            )
        else:
            # 让 Agent 决定本轮参数
            hyperparams = ask_llm_for_next_hyperparams(history_context, current_best_map50, current_best_map50_95, target_map50, target_map50_95)
            print(f"⚙️ Agent 下达的新参数配置: {hyperparams}")
            
            # 从 hyperparams 中移除可能重复的 imgsz 以免引起 dict 参数冲突
            if 'imgsz' in hyperparams:
                del hyperparams['imgsz']

            args = dict(
                model='yolov8-gold.yaml',
                data='norcardis_disease.yaml',
                device='0',
                batch=8,           # 移除了 BiFormer，恢复 batch 8
                imgsz=800,         # 恢复 800 高分辨率
                workers=4,         # 强制使用 4 进程加速
                mtl=1,
                cagrad=False,
                deterministic=False,
                resume=False,
                pretrained=True,   # 启用预训练加载
                pre_weights='yolo-fd.pt', # 加载 0.76 基线模型，自动兼容新架构
                epochs=150,
                amp=False,         # 禁用 AMP 以规避混合精度引发的死锁
                
                # --- 核心突破策略 ---
                freeze=10,         # 冻结 Backbone (前 10 层)。让 P2 头和 Ctxt 注意力先单独收敛，防止污染优秀的预训练特征提取器
                box=7.5,           # 默认是 7.5，适当增加定位 Loss 权重，压制过剩的分类 Loss
                dfl=2.0,           # 默认是 1.5，大幅增加 DFL 分布焦点损失权重，逼迫模型学习最精确的边界框，冲击 mAP50-95
                cls=0.5,           # 默认是 0.5，适当压低分类权重，因为 P2 头产生了极度不平衡的巨量负样本（背景）
                # --------------------
                **hyperparams
            )
        
        try:
            # 解决 Windows 下多进程 DataLoader 死锁问题
            import os
            os.environ['PIN_MEMORY'] = 'False'
            
            trainer = DetectionTrainer(overrides=args)
            trainer.train()
        except Exception as e:
            print(f"[ERROR] 训练崩溃: {e}")
            history_context += f"第{round_num}轮训练崩溃，原因: {e}\n"
            continue
            
        # 评估
        latest_dir = get_latest_run()
        round_map50, round_map50_95, tail_data = evaluate_run(latest_dir)
        
        print(f"\n📊 第 {round_num} 轮结束。跑分: mAP50 = {round_map50:.5f}, mAP50-95 = {round_map50_95:.5f}")
        
        # 更新历史上下文，供 Agent 下次分析
        history_context += f"第{round_num}轮 {'(续训)' if is_resuming_from_interruption and round_num == 1 else f'(参数: {json.dumps(hyperparams)})'}:\n"
        history_context += f"跑分: mAP50={round_map50:.5f}, mAP50-95={round_map50_95:.5f}\n"
        history_context += f"最后几轮 Loss 趋势:\n{tail_data}\n\n"
        
        # 记录全局最佳
        if round_map50 > current_best_map50: current_best_map50 = round_map50
        if round_map50_95 > current_best_map50_95: current_best_map50_95 = round_map50_95
        
        if round_map50 >= target_map50 and round_map50_95 >= target_map50_95:
            print(f"🎉 任务达成！大模型 Agent 成功在第 {round_num} 轮找到了最优解！")
            print(f"📦 达标模型权重保存在: {latest_dir}\\weights\\best.pt")
            break
        else:
            print("🔄 成绩未达标，正在将训练结果反馈给 Agent 筹备下一轮...")
            time.sleep(10)

if __name__ == "__main__":
    # Windows 多进程兼容性保护（非常重要，防止 workers=4 时无限循环或死锁）
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 全局禁用 OpenCV 多线程以防止和 PyTorch DataLoader 死锁冲突
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    run_llm_agent()
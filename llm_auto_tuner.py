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
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except:
        pass

def get_latest_run():
    runs_dir = os.path.join(project_dir, 'runs', 'detect')
    if not os.path.exists(runs_dir): return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs: return None
    return max(train_dirs, key=os.path.getmtime)

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

以下是历史实验和最近一轮的训练结果趋势：
{history_context}

请分析原因(是否过拟合、分辨率不够、增强是否太强)，并为下一轮训练提供一组新的超参数。
你必须严格返回合法的 JSON 格式数据，包含以下字段：
{{
    "analysis": "你的分析和理由(简短)",
    "hyperparameters": {{
        "imgsz": 800,  // 分辨率, 建议 640, 800 或 1024
        "mixup": 0.0,  // 0.0 到 0.2
        "copy_paste": 0.0, // 0.0 到 0.2
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
        response = requests.post(API_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # 适配 Volcengine Ark API 的返回格式
        response_data = response.json()
        
        # 针对不同可能的响应结构进行兼容处理
        content = ""
        if 'output' in response_data and isinstance(response_data['output'], dict):
            content = response_data['output'].get('text', '')
        elif 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice:
                content = choice['message'].get('content', '')
            elif 'text' in choice:
                content = choice.get('text', '')
                
        if not content:
            print(f"❌ 无法从响应中解析内容: {response_data}")
            return {"imgsz": 800, "mixup": 0.0, "copy_paste": 0.0, "pcgrad": False, "lr0": 0.005}
            
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
        return {"imgsz": 800, "mixup": 0.0, "copy_paste": 0.0, "pcgrad": False, "lr0": 0.005}


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
    
    for round_num in range(1, 11):  # 最多让 Agent 尝试 10 轮
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 准备第 {round_num} 轮训练...")
        
        # 让 Agent 决定本轮参数
        hyperparams = ask_llm_for_next_hyperparams(history_context, current_best_map50, current_best_map50_95, target_map50, target_map50_95)
        print(f"⚙️ Agent 下达的新参数配置: {hyperparams}")
        
        args = dict(
            model='yolov8-gold.yaml',
            data='norcardis_disease.yaml',
            device='0',
            batch=16,
            mtl=1,
            cagrad=False,
            deterministic=False,
            resume=False,
            pre_weights='yolo-fd.pt',
            epochs=150,
            **hyperparams
        )
        
        try:
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
        history_context += f"第{round_num}轮 (参数: {json.dumps(hyperparams)}):\n"
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

if __name__ == '__main__':
    run_llm_agent()
import os
import sys
import ctypes
import glob
import json
import time
import pandas as pd
import requests

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ultralytics.yolo.v8.detect import DetectionTrainer

API_KEY = os.getenv("ARK_API_KEY", "")
API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
MODEL_NAME = "doubao-seed-2-0-pro-260215"


def prevent_sleep():
    """Prevent Windows from sleeping during training."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001 | 0x00000002 | 0x00000040
        )
        print("[INFO] Anti-sleep mode enabled.")
    except Exception as e:
        print(f"[WARNING] Failed to set anti-sleep mode: {e}")


def get_latest_run():
    runs_dir = os.path.join(project_dir, "runs", "detect")
    if not os.path.exists(runs_dir):
        return None
    train_dirs = glob.glob(os.path.join(runs_dir, "train*"))
    if not train_dirs:
        return None
    return max(train_dirs, key=os.path.getmtime)


def get_last_interrupted_run():
    runs_dir = os.path.join(project_dir, "runs", "detect")
    if not os.path.exists(runs_dir):
        return None
    train_dirs = glob.glob(os.path.join(runs_dir, "train*"))
    if not train_dirs:
        return None
    latest_dir = max(train_dirs, key=os.path.getmtime)
    last_pt = os.path.join(latest_dir, "weights", "last.pt")
    return last_pt if os.path.exists(last_pt) else None


def evaluate_run(latest_dir):
    if not latest_dir:
        return 0.0, 0.0, ""
    results_file = os.path.join(latest_dir, "results.csv")
    if not os.path.exists(results_file):
        return 0.0, 0.0, ""

    try:
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        map50_col = [c for c in df.columns if "mAP50(B)" in c][0]
        map50_95_col = [c for c in df.columns if "mAP50-95(B)" in c][0]
        best_map50 = float(df[map50_col].max())
        best_map50_95 = float(df[map50_95_col].max())
        tail_data = df[
            [
                "epoch",
                "train/box_loss",
                "train/cls_loss",
                "val/box_loss",
                "val/cls_loss",
                map50_col,
                map50_95_col,
            ]
        ].tail(5).to_string()
        return best_map50, best_map50_95, tail_data
    except Exception as e:
        print(f"[ERROR] Failed to parse results.csv: {e}")
        return 0.0, 0.0, ""


def ask_llm_for_next_hyperparams(history_context, current_best_map50, current_best_map50_95, target_map50, target_map50_95):
    """Call LLM and return next-round hyperparameters."""
    default_params = {
        "mixup": 0.1,
        "copy_paste": 0.1,
        "pcgrad": False,
        "lr0": 0.005,
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "weight_decay": 0.0005,
    }

    if not API_KEY:
        print("[WARNING] ARK_API_KEY is not set. Using default hyperparameters.")
        return default_params

    prompt = f"""
You are a YOLO multi-task tuning assistant.
Current best:
- mAP50: {current_best_map50}
- mAP50-95: {current_best_map50_95}
Target:
- mAP50 >= {target_map50}
- mAP50-95 >= {target_map50_95}

History:
{history_context}

Return valid JSON only:
{{
  "analysis": "short reason",
  "hyperparameters": {{
    "mixup": 0.1,
    "copy_paste": 0.1,
    "lr0": 0.005,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "weight_decay": 0.0005,
    "pcgrad": false
  }}
}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }

    try:
        print("[INFO] Requesting LLM for next hyperparameters...")
        response = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        content = ""
        if "output" in data:
            if isinstance(data["output"], dict):
                content = data["output"].get("text", "")
            elif isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "text":
                        content = item.get("text", "")
                        break
        elif "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content", "")
            elif "text" in choice:
                content = choice.get("text", "")

        if not content:
            print(f"[ERROR] Empty LLM response: {data}")
            return default_params

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content)
        print(f"[INFO] LLM analysis: {result.get('analysis')}")
        return result.get("hyperparameters", default_params)
    except Exception as e:
        print(f"[ERROR] LLM request failed: {e}")
        return default_params


def run_llm_agent():
    prevent_sleep()
    target_map50, target_map50_95 = 0.96, 0.80

    print("===========================================================")
    print("[INFO] YOLO-FD LLM Auto-Tuner started")
    print(f"[INFO] Target: mAP50>={target_map50}, mAP50-95>={target_map50_95}")
    print("===========================================================")

    history_context = "Round0 baseline: mAP50=0.933, mAP50-95=0.760\n"
    current_best_map50 = 0.933
    current_best_map50_95 = 0.760

    last_pt_path = get_last_interrupted_run()
    if last_pt_path:
        print(f"[INFO] Found interrupted checkpoint: {last_pt_path}")

    for round_num in range(1, 11):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Round {round_num}/10")

        hyperparams = ask_llm_for_next_hyperparams(
            history_context,
            current_best_map50,
            current_best_map50_95,
            target_map50,
            target_map50_95,
        )
        print(f"[INFO] hyperparams: {hyperparams}")

        if "imgsz" in hyperparams:
            del hyperparams["imgsz"]

        args = dict(
            model="yolov8-gold.yaml",
            data="norcardis_disease.yaml",
            device="0",
            batch=8,
            imgsz=800,
            workers=4,
            mtl=1,
            cagrad=False,
            deterministic=False,
            resume=False,
            pretrained=True,
            pre_weights="yolo-fd.pt",
            epochs=150,
            amp=False,
            freeze=10,
            box=7.5,
            dfl=2.0,
            cls=0.5,
            **hyperparams,
        )

        try:
            os.environ["PIN_MEMORY"] = "False"
            trainer = DetectionTrainer(overrides=args)
            trainer.train()
        except Exception as e:
            print(f"[ERROR] Training crashed: {e}")
            history_context += f"Round {round_num}: failed, reason={e}\n"
            continue

        latest_dir = get_latest_run()
        round_map50, round_map50_95, tail_data = evaluate_run(latest_dir)
        print(f"[INFO] Round {round_num} metrics: mAP50={round_map50:.5f}, mAP50-95={round_map50_95:.5f}")

        history_context += f"Round {round_num}: mAP50={round_map50:.5f}, mAP50-95={round_map50_95:.5f}\n"
        history_context += f"Tail:\n{tail_data}\n\n"

        current_best_map50 = max(current_best_map50, round_map50)
        current_best_map50_95 = max(current_best_map50_95, round_map50_95)

        if round_map50 >= target_map50 and round_map50_95 >= target_map50_95:
            print(f"[INFO] Target reached at round {round_num}.")
            print(f"[INFO] Best model path: {latest_dir}\\weights\\best.pt")
            break

        print("[INFO] Not reached target, continue next round...")
        time.sleep(10)


if __name__ == "__main__":
    import multiprocessing
    import cv2

    multiprocessing.freeze_support()
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    run_llm_agent()

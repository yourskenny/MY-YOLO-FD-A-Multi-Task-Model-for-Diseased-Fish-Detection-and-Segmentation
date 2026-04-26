import sys
import os

project_dir = r"C:\coding\YOLO-FD\YOLO-FD-main (1)(1)\YOLO-FD-main"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from ultralytics.yolo.engine.model import YOLO
import torch

try:
    print("Testing YOLO-FD custom model parsing...")
    model = YOLO("ultralytics/models/v8/yolov8-gold.yaml")
    
    # Just print the model info
    model.info()
    
    # Let's try a dummy forward pass to see if tensor shapes match!
    # Model input shape should be [B, C, H, W]
    dummy_input = torch.zeros((1, 3, 640, 640))
    print("Running dummy forward pass...")
    out = model.model(dummy_input)
    print("Forward pass successful!")
    print("Outputs:", [o.shape if hasattr(o, 'shape') else len(o) for o in out])
    print("ALL TESTS PASSED!")
except Exception as e:
    import traceback
    traceback.print_exc()

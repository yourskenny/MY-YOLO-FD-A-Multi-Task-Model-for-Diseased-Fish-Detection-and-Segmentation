from ultralytics import YOLO
import time
import cv2
import os

def main():
    model_path = "yolo-fd.pt"
    img_path = "ultralytics/assets/bus.jpg"
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found!")
        return
        
    print(f"Loading PyTorch model {model_path}...")
    model = YOLO(model_path)
    
    data_path = "ultralytics/datasets/norcardis_disease.yaml"
    
    print("Testing PyTorch inference speed...")
    for _ in range(3):
        _ = model(img_path, data=data_path, verbose=False)
        
    start_time = time.time()
    for _ in range(10):
        _ = model(img_path, data=data_path, verbose=False)
    pt_time = (time.time() - start_time) / 10
    print(f"PyTorch avg inference time: {pt_time*1000:.2f} ms")

    print("Testing PyTorch FP16 inference speed...")
    for _ in range(3):
        _ = model(img_path, data=data_path, half=True, verbose=False)
        
    start_time = time.time()
    for _ in range(10):
        _ = model(img_path, data=data_path, half=True, verbose=False)
    pt_half_time = (time.time() - start_time) / 10
    print(f"PyTorch FP16 avg inference time: {pt_half_time*1000:.2f} ms")
    
    print("Exporting model to ONNX format...")
    onnx_path = model.export(format="onnx", simplify=True)
    print(f"Model exported to {onnx_path}")
    
    print("Loading ONNX model...")
    onnx_model = YOLO(onnx_path, task="detect")
    
    print("Testing ONNX inference speed...")
    for _ in range(3):
        _ = onnx_model(img_path, data=data_path, verbose=False)
        
    start_time = time.time()
    for _ in range(10):
        _ = onnx_model(img_path, data=data_path, verbose=False)
    onnx_time = (time.time() - start_time) / 10
    print(f"ONNX avg inference time: {onnx_time*1000:.2f} ms")
    
    print("--- Speed Comparison ---")
    print(f"PyTorch FP32: {pt_time*1000:.2f} ms")
    print(f"PyTorch FP16: {pt_half_time*1000:.2f} ms")
    print(f"ONNX: {onnx_time*1000:.2f} ms")
    if pt_time > 0:
        print(f"Speedup: {pt_time / onnx_time:.2f}x")

if __name__ == "__main__":
    main()

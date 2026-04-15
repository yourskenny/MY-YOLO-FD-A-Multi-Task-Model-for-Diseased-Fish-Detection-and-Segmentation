from ultralytics.yolo.v8.detect import DetectionValidator
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG


def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = 'norcardis_disease.yaml'
    model = './yolo-fd.pt'
    batch = 8   # cfg.batch * 2
    args = dict(model=model, data=data, batch=batch, mode="val", imgsz=640, device="cpu")
    args = get_cfg(cfg, overrides=args)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        metrics = validator(model=args.model)
        print("FINAL METRICS:", metrics)


if __name__ == '__main__':
    val()
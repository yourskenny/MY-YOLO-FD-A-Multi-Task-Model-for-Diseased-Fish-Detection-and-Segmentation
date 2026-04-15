from ultralytics import YOLO

model = YOLO('yolo-fd.pt')
results = model.val(data='norcardis_disease.yaml', device='cpu')

print(results.results_dict)

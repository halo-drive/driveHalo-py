from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./yolov8n-seg.pt')  # Path to your YOLOv8 model file

# Export to ONNX with dynamic axes
model.export(format='onnx', dynamic=True, imgsz=640)

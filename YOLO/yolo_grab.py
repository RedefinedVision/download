import argparse
from ultralytics import YOLO

def get_user_input():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
    parser.add_argument('--model_name', type=str, default='yolov8m', choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], help='YOLO model name')
    parser.add_argument('--input_width', type=int, default=320, help='Input width')
    parser.add_argument('--input_height', type=int, default=180, help='Input height')
    parser.add_argument('--optimize_cpu', action='store_true', help='Optimize for CPU')
    args = parser.parse_args()
    return args

args = get_user_input()

model = YOLO(f"{args.model_name}.pt")
model.export(format="onnx", imgsz=[args.input_height, args.input_width], optimize=args.optimize_cpu)

print("Model exported successfully!")

import os
from ultralytics import YOLO
import torch

def train_yolov8():
    """Train a YOLOv8 model for plate detection."""
    model = YOLO("yolov8n.yaml") #specifies the architecture of the YOLOv8 model
    results = model.train(
        data="replace/with/your/platev8.yaml", #datasetconfiguration
        epochs=100,
        imgsz=640,
        batch=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr0=0.005,
        augment=True,
        iou=0.5
    )
    return results

def validate_yolov8():
    """Validate the trained YOLOv8 model."""
    model = YOLO("runs/detect/train/weights/best.pt")
    metrics = model.val(data="replace/with/your/platev8.yaml")
    print(f"mAP@50:{metrics.box.map50}")
    print(f"mAP@75:{metrics.box.map75}")
    print(f"mAP@50-95:{metrics.box.maps}")
    return metrics

if __name__ == "__main__":
    train_yolov8()
    validate_yolov8()

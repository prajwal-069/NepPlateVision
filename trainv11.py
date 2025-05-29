import os
from ultralytics import YOLO
import torch

def train_yolov11():
    """Train a YOLOv11 model for character detection."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model = YOLO("yolo11n.yaml") #specifies the architecture of the YOLOv11 model
    results = model.train(
        data="replace/with/your/characterv11.yaml" #datasetconfiguration,
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.005,
        augment=True,
        amp=True,
        iou=0.5
    )
    return results

def validate_yolov11():
    """Validate the trained YOLOv11 model."""
    model = YOLO("runs/detect/train2/weights/best.pt")
    metrics = model.val(data="replace/with/your/characterv11.yaml")
    print(f"mAP@50:{metrics.box.map50}")
    print(f"mAP@75:{metrics.box.map75}")
    print(f"mAP@50-95:{metrics.box.maps}")
    return metrics

if __name__ == "__main__":
    train_yolov11()
    validate_yolov11()

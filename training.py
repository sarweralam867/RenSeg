from ultralytics import YOLO
import torch

def train_model(data_yaml, epochs=80, batch_size=16, imgsz=640, save_dir="checkpoints"):
    # Disable cuDNN to avoid potential issues
    torch.backends.cudnn.enabled = False

    # Initialize YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=save_dir
    )

    return model

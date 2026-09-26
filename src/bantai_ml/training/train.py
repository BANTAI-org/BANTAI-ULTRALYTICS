from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def train_model(
    model: str = "yolov8n.pt",
    data: str | Path = "configs/data/bantai.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    project: str = "runs",
    name: str = "bantai-baseline",
    exist_ok: bool = True,
    **kwargs,
):
    """Train a YOLOv8 model for the BANTAI dataset."""
    model_obj = YOLO(model)
    return model_obj.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        **kwargs,
    )


if __name__ == "__main__":
    train_model()

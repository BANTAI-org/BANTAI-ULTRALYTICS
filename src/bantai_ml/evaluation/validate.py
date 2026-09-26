from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def validate_model(model: str | YOLO, data: str | Path = "configs/data/bantai.yaml"):
    """Run model validation on the configured dataset."""
    model_obj = model if isinstance(model, YOLO) else YOLO(model)
    return model_obj.val(data=str(data))


if __name__ == "__main__":
    validate_model("runs/train/best.pt")

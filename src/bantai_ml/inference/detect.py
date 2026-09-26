from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def detect_objects(
    model: str | YOLO,
    source: str | Path,
    project: str = "runs/predict",
    name: str = "exp",
    exist_ok: bool = True,
    **kwargs,
):
    """Run object detection inference over a source image or folder."""
    model_obj = model if isinstance(model, YOLO) else YOLO(model)
    return model_obj.predict(
        source=str(source),
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        **kwargs,
    )


if __name__ == "__main__":
    detect_objects("runs/train/weights/best.pt", "data/raw")

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def track_objects(
    model: str | YOLO,
    source: str | Path,
    project: str = "runs/track",
    name: str = "exp",
    exist_ok: bool = True,
    **kwargs,
):
    """Run object tracking inference over a source video or sequence."""
    model_obj = model if isinstance(model, YOLO) else YOLO(model)
    return model_obj.track(
        source=str(source),
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        **kwargs,
    )


if __name__ == "__main__":
    track_objects("runs/train/weights/best.pt", "data/raw/video.mp4")

from __future__ import annotations

from pathlib import Path


def validate_dataset(data_dir: str | Path = "data/processed/bantai") -> dict[str, int]:
    """Check that the expected train/val/test image dirs exist and count files."""
    data_path = Path(data_dir)
    split_dirs = ["train", "val", "test"]
    counts: dict[str, int] = {}

    for split in split_dirs:
        image_dir = data_path / "images" / split
        label_dir = data_path / "labels" / split

        image_count = len(list(image_dir.glob("*"))) if image_dir.exists() else 0
        label_count = len(list(label_dir.glob("*"))) if label_dir.exists() else 0
        counts[split] = {"images": image_count, "labels": label_count}

    return counts


if __name__ == "__main__":
    print(validate_dataset())

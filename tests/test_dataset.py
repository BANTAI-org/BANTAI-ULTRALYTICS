from pathlib import Path

from bantai_ml.data.validate_dataset import validate_dataset


def test_validate_dataset_returns_dict():
    result = validate_dataset(Path("data/processed/bantai"))
    assert isinstance(result, dict)
    assert set(result.keys()) == {"train", "val", "test"}

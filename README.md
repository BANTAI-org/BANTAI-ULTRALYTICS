# BANTAI Ultralytics

Machine learning scaffolding for object detection training and inference with Ultralytics YOLO.

## Project structure

- `configs/` – dataset and training configuration files
- `src/bantai_ml/` – Python package for training, validation, inference, and data checks
- `scripts/` – command-line entry points
- `data/` – raw, interim, and processed dataset storage
- `models/` – pretrained and checkpoint weights
- `runs/` – training and prediction artifacts
- `tests/` – project tests

## Quick start

1. Create a virtual environment and activate it.
2. Install dependencies:
   `python -m pip install -r requirements.txt`
3. Train a model:
   `python scripts/train.py`
4. Validate a model:
   `python scripts/validate.py`
5. Run inference:
   `python scripts/predict.py`

## Notes

The default dataset config expects a YOLO-style layout under `data/processed/bantai`:

- `images/train`
- `images/val`
- `images/test`
- `labels/train`
- `labels/val`
- `labels/test`

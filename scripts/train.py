import sys
from pathlib import Path

# Add src to path so we can import bantai_ml
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bantai_ml.training.train import train_model


if __name__ == "__main__":
    train_model(
        data="data/processed/guns/guns.yaml",
        name="guns-baseline",
        epochs=150,
        batch=8,
        device=0 # Using CPU until CUDA PyTorch is installed
    )

"""
Training script for combined weapons detection (guns + knives).
Uses enhanced configuration for yolo26s with multi-class detection.
"""

from pathlib import Path
from ultralytics import YOLO

def train_weapons():
    """Train with combined weapons dataset."""
    model = YOLO("yolo26s.pt")
    
    results = model.train(
        data="configs/data/weapons.yaml",
        cfg="configs/train/enhanced.yaml",
        device=0,  # Use GPU 0
        workers=8,
        verbose=True,
        name="weapons-enhanced",
    )
    
    return results

if __name__ == "__main__":
    print("Starting enhanced training for weapons detection (guns + knives)...")
    print("Model: yolo26s (optimized for multi-stream inference)")
    print("Classes: gun, knife")
    print("Configuration: configs/train/enhanced.yaml")
    print("-" * 60)
    
    results = train_weapons()
    
    print("\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")

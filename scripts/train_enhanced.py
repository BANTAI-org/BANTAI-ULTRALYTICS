"""
Enhanced training script for BANTAI weapons detection (3-class: Gun, Knife, Pistol).
Uses aggressive augmentation and optimized hyperparameters for yolo26s.
Dataset: guns.v1i + Knife.v2i merged, grenades removed (1,737 training images)
"""

from pathlib import Path
from ultralytics import YOLO

def train_enhanced():
    """Train with enhanced configuration."""
    model = YOLO("yolo26s.pt")
    
    results = model.train(
        data="C:/BANTAI-DATA/processed/weapons-no-grenades/data.yaml",
        cfg="configs/train/enhanced.yaml",
        device=0,  # Use GPU 0
        workers=8,
        verbose=True,
        name="weapons-3class-enhanced",
    )
    
    return results

if __name__ == "__main__":
    print("Starting enhanced training for BANTAI weapons detection...")
    print("Model: yolo26s (optimized for multi-stream inference)")
    print("Dataset: Merged weapons (guns.v1i + Knife.v2i, no grenades)")
    print("Training images: 1,737 | Validation: 311")
    print("Classes: Gun, Knife, Pistol (3 classes)")
    print("Configuration: configs/train/enhanced.yaml")
    print("-" * 60)
    
    results = train_enhanced()
    
    print("\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")

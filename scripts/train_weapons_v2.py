"""
Training script for combined weapons detection v2 (expanded dataset).
Includes multiple gun datasets + knives with 4 classes total.
"""

from pathlib import Path
from ultralytics import YOLO

def train_weapons_v2():
    """Train with expanded weapons dataset."""
    model = YOLO("yolo26s.pt")
    
    results = model.train(
        data="configs/data/weapons-v2.yaml",
        cfg="configs/train/enhanced.yaml",
        device=0,
        workers=4,  # Reduced workers to avoid CUDA issues
        batch=8,    # Reduced batch size for stability
        verbose=True,
        name="weapons-v2-enhanced",
    )
    
    return results

if __name__ == "__main__":
    print("Starting enhanced training for weapons detection v2...")
    print("Model: yolo26s (optimized for multi-stream inference)")
    print("Classes: gun, pistol, knife, grenade")
    print("Training images: ~2,316 (combined datasets)")
    print("Configuration: configs/train/enhanced.yaml")
    print("-" * 60)
    
    results = train_weapons_v2()
    
    print("\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")

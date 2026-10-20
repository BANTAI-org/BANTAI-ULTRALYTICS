"""
Enhanced training for 3-class weapons detection with focus on knife class.
Uses data augmentation optimized for low-quality camera conditions.
"""

from pathlib import Path
from ultralytics import YOLO

def train_weapons_3class():
    """Train 3-class model with enhanced knife detection."""
    model = YOLO("yolo26s.pt")
    
    results = model.train(
        data="C:/BANTAI-DATA/processed/weapons-no-grenades/data.yaml",
        cfg="configs/train/enhanced.yaml",
        device=0,
        workers=8,
        epochs=150,
        batch=8,
        imgsz=640,
        
        # Enhanced augmentation for better generalization to low-quality cameras
        hsv_h=0.015,      # Hue augmentation (lighting variations)
        hsv_s=0.7,        # Saturation (camera quality variations)
        hsv_v=0.4,        # Value/brightness (poor lighting)
        degrees=10,       # Rotation (different angles)
        translate=0.2,    # Translation (off-center detections)
        scale=0.5,        # Scale (different distances)
        flipud=0.0,       # No vertical flip (weapons don't flip)
        fliplr=0.5,       # Horizontal flip (left/right hand)
        mosaic=1.0,       # Mosaic augmentation (multiple objects)
        mixup=0.1,        # Mixup (helps with occlusion)
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,        # Initial learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights - emphasize knife class
        box=7.5,          # Box loss weight
        cls=0.5,          # Class loss weight
        
        verbose=True,
        name="weapons-3class-enhanced-v2",
        
        # Enable multi-scale training
        rect=False,       # Rectangular training disabled for better augmentation
        cache=True,       # Cache images for faster training
    )
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING 3-CLASS WEAPONS DETECTOR")
    print("=" * 60)
    print("Model: yolo26s (optimized for multi-stream inference)")
    print("Dataset: Merged weapons (no grenades)")
    print("Training images: 1,737 | Validation: 311")
    print("Classes: Gun, Knife, Pistol")
    print()
    print("ENHANCEMENTS:")
    print("  • Aggressive augmentation for low-quality cameras")
    print("  • HSV augmentation for lighting/quality variations")
    print("  • Scale augmentation for different distances")
    print("  • Mosaic + Mixup for complex scenes")
    print("=" * 60)
    print()
    
    results = train_weapons_3class()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print()
    print("Test with: python scripts/test_camera.py --model " + 
          f"{results.save_dir}/weights/best.pt")

"""
Merge guns.v1i.yolo26 and Knife.v2i.yolo26 datasets.
Remaps knife labels from the knife-only dataset to match the guns dataset class indices.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Class mapping for guns.v1i.yolo26
# 0: Grenade, 1: Gun, 2: Knife, 3: Pistol
KNIFE_CLASS_INDEX = 2

def remap_knife_labels(label_file, output_file):
    """Remap knife labels from class 0 to class 2."""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            # Change class index from 0 to 2 (knife)
            parts[0] = str(KNIFE_CLASS_INDEX)
            remapped_lines.append(' '.join(parts) + '\n')
    
    with open(output_file, 'w') as f:
        f.writelines(remapped_lines)

def merge_datasets():
    """Merge guns and knife datasets."""
    
    # Source paths
    guns_base = Path("C:/Users/Kurt Russel/Downloads/guns.v1i.yolo26")
    knife_base = Path("C:/Users/Kurt Russel/Downloads/Knife.v2i.yolo26")
    
    # Output path
    merged_base = Path("C:/BANTAI-DATA/processed/weapons-merged")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (merged_base / split / 'images').mkdir(parents=True, exist_ok=True)
        (merged_base / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MERGING DATASETS")
    print("=" * 60)
    
    # Copy guns dataset first (all splits)
    print("\n[1/3] Copying guns dataset...")
    for split in ['train', 'valid', 'test']:
        src_images = guns_base / split / 'images'
        src_labels = guns_base / split / 'labels'
        
        if src_images.exists():
            images = list(src_images.glob('*.jpg'))
            print(f"  {split}: {len(images)} images")
            
            for img in tqdm(images, desc=f"  Copying {split}"):
                # Copy image
                shutil.copy2(img, merged_base / split / 'images' / img.name)
                
                # Copy label
                label = src_labels / f"{img.stem}.txt"
                if label.exists():
                    shutil.copy2(label, merged_base / split / 'labels' / label.name)
    
    # Add knife dataset (train only, as it doesn't have valid/test splits)
    print("\n[2/3] Adding knife dataset to training set...")
    knife_images = list((knife_base / 'train' / 'images').glob('*.jpg'))
    print(f"  train: {len(knife_images)} knife images")
    
    for img in tqdm(knife_images, desc="  Processing knives"):
        # Copy image with prefix to avoid name collisions
        new_name = f"knife_{img.name}"
        shutil.copy2(img, merged_base / 'train' / 'images' / new_name)
        
        # Remap and copy label
        label = knife_base / 'train' / 'labels' / f"{img.stem}.txt"
        if label.exists():
            output_label = merged_base / 'train' / 'labels' / f"knife_{img.stem}.txt"
            remap_knife_labels(label, output_label)
    
    # Create data.yaml
    print("\n[3/3] Creating data.yaml...")
    data_yaml = merged_base / 'data.yaml'
    yaml_content = f"""# Merged weapons dataset
path: {str(merged_base).replace(chr(92), '/')}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: Grenade
  1: Gun
  2: Knife
  3: Pistol

# Dataset composition:
# - guns.v1i.yolo26: Original 4-class weapons dataset
# - Knife.v2i.yolo26: Additional 700 knife images (remapped to class 2)
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Data YAML saved: {data_yaml}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        img_count = len(list((merged_base / split / 'images').glob('*.jpg')))
        lbl_count = len(list((merged_base / split / 'labels').glob('*.txt')))
        print(f"{split:6s}: {img_count:4d} images, {lbl_count:4d} labels")
    
    print(f"\nMerged dataset: {merged_base}")
    print(f"Data YAML: {data_yaml}")
    print("\nReady to train with enhanced knife detection!")

if __name__ == "__main__":
    merge_datasets()

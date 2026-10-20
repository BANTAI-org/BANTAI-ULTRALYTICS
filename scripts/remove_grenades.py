"""
Remove grenade class from weapons-merged dataset.
Creates new dataset with only Gun (0), Knife (1), Pistol (2).
Remaps class indices: Gun (1->0), Knife (2->1), Pistol (3->2)
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def has_grenade(label_file):
    """Check if label file contains grenade (class 0)."""
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == 0:  # Class 0 = Grenade
                return True
    return False

def remap_labels(label_file, output_file):
    """
    Remove grenade detections and remap class indices.
    Old: 0=Grenade, 1=Gun, 2=Knife, 3=Pistol
    New: 0=Gun, 1=Knife, 2=Pistol
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            old_class = int(parts[0])
            
            # Skip grenades (class 0)
            if old_class == 0:
                continue
            
            # Remap: 1->0 (Gun), 2->1 (Knife), 3->2 (Pistol)
            new_class = old_class - 1
            parts[0] = str(new_class)
            remapped_lines.append(' '.join(parts) + '\n')
    
    # Only write file if it has detections after removing grenades
    if remapped_lines:
        with open(output_file, 'w') as f:
            f.writelines(remapped_lines)
        return True
    return False

def remove_grenades():
    """Remove grenades and create new 3-class dataset."""
    
    # Source and destination
    source_base = Path("C:/BANTAI-DATA/processed/weapons-merged")
    output_base = Path("C:/BANTAI-DATA/processed/weapons-no-grenades")
    
    print("=" * 60)
    print("REMOVING GRENADES FROM DATASET")
    print("=" * 60)
    print(f"Source: {source_base}")
    print(f"Output: {output_base}")
    print()
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_base / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_base / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train': {'total': 0, 'kept': 0, 'removed_grenade': 0, 'removed_empty': 0},
        'valid': {'total': 0, 'kept': 0, 'removed_grenade': 0, 'removed_empty': 0},
        'test': {'total': 0, 'kept': 0, 'removed_grenade': 0, 'removed_empty': 0}
    }
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        src_images = source_base / split / 'images'
        src_labels = source_base / split / 'labels'
        
        if not src_images.exists():
            continue
        
        images = list(src_images.glob('*.jpg'))
        print(f"\n[Processing {split}] {len(images)} images...")
        
        for img in tqdm(images, desc=f"  {split}"):
            stats[split]['total'] += 1
            label = src_labels / f"{img.stem}.txt"
            
            if not label.exists():
                # No label file - skip
                stats[split]['removed_empty'] += 1
                continue
            
            # Check if image contains ONLY grenades
            has_only_grenades = True
            with open(label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) != 0:  # Has non-grenade class
                        has_only_grenades = False
                        break
            
            if has_only_grenades:
                # Skip images with only grenades
                stats[split]['removed_grenade'] += 1
                continue
            
            # Remap labels (remove grenades, remap other classes)
            output_label = output_base / split / 'labels' / label.name
            has_detections = remap_labels(label, output_label)
            
            if has_detections:
                # Copy image
                shutil.copy2(img, output_base / split / 'images' / img.name)
                stats[split]['kept'] += 1
            else:
                # After removing grenades, no detections left
                stats[split]['removed_empty'] += 1
    
    # Create data.yaml
    print("\n[Creating data.yaml]")
    data_yaml = output_base / 'data.yaml'
    yaml_content = f"""# Weapons dataset without grenades (3 classes)
path: {str(output_base).replace(chr(92), '/')}
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: Gun
  1: Knife
  2: Pistol

# Grenades removed from original weapons-merged dataset
# Class indices remapped: Gun (1->0), Knife (2->1), Pistol (3->2)
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"Data YAML saved: {data_yaml}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("REMOVAL COMPLETE!")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        s = stats[split]
        if s['total'] > 0:
            print(f"\n{split.upper()}:")
            print(f"  Original images:     {s['total']}")
            print(f"  Kept:                {s['kept']}")
            print(f"  Removed (grenades):  {s['removed_grenade']}")
            print(f"  Removed (empty):     {s['removed_empty']}")
            print(f"  Retention rate:      {s['kept']/s['total']*100:.1f}%")
    
    # Overall stats
    total_kept = sum(s['kept'] for s in stats.values())
    total_original = sum(s['total'] for s in stats.values())
    
    print(f"\nOVERALL:")
    print(f"  Total images kept:   {total_kept}")
    print(f"  Total images removed: {total_original - total_kept}")
    print(f"\nNew dataset: {output_base}")
    print(f"Data YAML: {data_yaml}")
    print("\nReady to train with 3-class detection (Gun, Knife, Pistol)!")

if __name__ == "__main__":
    remove_grenades()

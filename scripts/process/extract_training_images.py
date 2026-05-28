#!/usr/bin/env python3
"""
Extract RGB images for hand and bags2 object.
"""
import os
import shutil
import h5py
import numpy as np
from PIL import Image
from pathlib import Path

dataset_root = "/media/rana/Balthazar/3Dreconstruction/manus/dataset/subject0"
output_root = "/media/rana/Balthazar/3Dreconstruction/manus/training_images"

hand_dir = os.path.join(output_root, "hand")
bags2_dir = os.path.join(output_root, "bags2")
os.makedirs(hand_dir, exist_ok=True)
os.makedirs(bags2_dir, exist_ok=True)

print("=" * 60)
print("Extracting Training Images")
print("=" * 60)

# 1. Extract hand images from HDF5
print(f"\n[1] Extracting hand images from HDF5...")
hdf5_dir = os.path.join(dataset_root, "actions_hdf5")
hand_count = 0

def extract_datasets_recursively(h5_obj, action_name, prefix=""):
    global hand_count
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset):
            # Only process image-like data: 3D (H,W,C) or 4D (N,H,W,C)
            if len(item.shape) == 3 and item.shape[2] in [3, 4]:
                # Single image: (H, W, C)
                data = item[:]
                if data.dtype == np.uint8:
                    arr = data
                elif data.dtype in [np.float32, np.float64]:
                    if data.max() <= 1.0:
                        arr = (data * 255).astype(np.uint8)
                    else:
                        arr = np.clip(data, 0, 255).astype(np.uint8)
                else:
                    arr = None
                
                if arr is not None:
                    try:
                        img = Image.fromarray(arr)
                        fname = f"{action_name}_{key}.png"
                        img.save(os.path.join(hand_dir, fname))
                        hand_count += 1
                    except:
                        pass
            elif len(item.shape) == 4 and item.shape[3] in [3, 4]:
                # Frame stack: (N, H, W, C)
                for i in range(item.shape[0]):
                    data = item[i]
                    if data.dtype == np.uint8:
                        arr = data
                    elif data.dtype in [np.float32, np.float64]:
                        if data.max() <= 1.0:
                            arr = (data * 255).astype(np.uint8)
                        else:
                            arr = np.clip(data, 0, 255).astype(np.uint8)
                    else:
                        arr = None
                    
                    if arr is not None:
                        try:
                            img = Image.fromarray(arr)
                            fname = f"{action_name}_{key}_{i:04d}.png"
                            img.save(os.path.join(hand_dir, fname))
                            hand_count += 1
                        except:
                            pass
        elif isinstance(item, h5py.Group):
            extract_datasets_recursively(item, action_name, prefix + "  ")

for hdf5_file in sorted(os.listdir(hdf5_dir)):
    if hdf5_file.endswith(".hdf5"):
        action = hdf5_file.replace(".hdf5", "")
        print(f"  Processing: {action}")
        with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'r') as f:
            extract_datasets_recursively(f, action)

print(f"  Extracted: {hand_count} hand images")

# 2. Copy bags2 images
print(f"\n[2] Copying bags2 images...")
bags2_src = os.path.join(dataset_root, "objects", "bags2", "images", "image")
bags2_count = 0

for root, dirs, files in os.walk(bags2_src):
    for file in files:
        src = os.path.join(root, file)
        rel = os.path.relpath(src, bags2_src)
        dst = os.path.join(bags2_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        bags2_count += 1

print(f"  Copied: {bags2_count} bags2 images")

print("\n" + "=" * 60)
print(f"✓ Done! Images at: {output_root}")
print("=" * 60)

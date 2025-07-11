from data_preparation import *
from fine_tune_model import *  

import os
import random
import cv2
import torch
import torch.amp
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.nn.utils
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
import sam2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import gc
import psutil
import time

def monitor_resources():
    """Monitor GPU and CPU usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached")
    
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    print(f"CPU: {cpu_percent}%, RAM: {memory_percent}%")

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def read_image(image_path, mask_path):  # read and resize image and mask
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    
    # Reduce image size to prevent memory issues
    r = np.min([512 / img.shape[1], 512 / img.shape[0]])  # Reduced from 1024 to 512
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.array([])
    
    for i in range(min(num_points, len(coords))):  # Ensure we don't exceed available points
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

print("Starting inference with resource monitoring...")
monitor_resources()

# Randomly select a test image from the test_data
selected_entry = random.choice(test_data)
print(f"Selected image: {selected_entry}")
image_path = selected_entry['image']
mask_path = selected_entry['annotation']
print(f"Mask path: {mask_path}")

# Load the selected image and mask
print("Loading and resizing image...")
image, target_mask = read_image(image_path, mask_path)
print(f"Image shape: {image.shape}, Mask shape: {target_mask.shape}")

# Generate random points for the input (reduced number)
num_samples = 10  # Reduced from 30 to 10 to save memory
input_points = get_points(target_mask, num_samples)
print(f"Generated {len(input_points)} input points")

if len(input_points) == 0:
    print("No valid points found in mask!")
    exit()

# Clear cache before loading model
clear_gpu_cache()
monitor_resources()

# Load the fine-tuned model
print("Loading fine-tuned model...")
FINE_TUNED_MODEL_WEIGHTS = "./fine_tuned_models/cork_analizer_sam2_final.pt"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

# Clear cache after loading model
clear_gpu_cache()
monitor_resources()

# Perform inference and predict masks
print("Starting inference...")
with torch.no_grad():
    with autocast("cpu"):  # Use mixed precision for inference
        predictor.set_image(image)
        
        # Add small delay to prevent overheating
        time.sleep(0.1)
        
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
        
        # Clear cache after prediction
        clear_gpu_cache()

print(f"Inference complete. Generated {len(masks)} masks")
monitor_resources()

# Process the predicted masks and sort by scores
np_masks = np.array(masks[:, 0])
np_scores = scores[:, 0]
sorted_masks = np_masks[np.argsort(np_scores)][::-1]

print("Processing masks...")
# Initialize segmentation map and occupancy mask
seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

# Combine masks to create the final segmentation map
for i in range(sorted_masks.shape[0]):
    mask = sorted_masks[i]
    if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
        continue

    mask_bool = mask.astype(bool)
    mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
    seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
    occupancy_mask[mask_bool] = True  # Update occupancy_mask

# Clear cache before visualization
clear_gpu_cache()
monitor_resources()

print("Creating visualization...")
# Visualization: Show the original image, mask, and final segmentation side by side
plt.figure(figsize=(12, 4))  # Reduced figure size

plt.subplot(1, 3, 1)
plt.title('Test Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Original Mask')
plt.imshow(target_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Final Segmentation')
plt.imshow(seg_map, cmap='jet')
plt.axis('off')

plt.tight_layout()
plt.show()

# Final cleanup
clear_gpu_cache()
print("Inference completed successfully!")
monitor_resources()
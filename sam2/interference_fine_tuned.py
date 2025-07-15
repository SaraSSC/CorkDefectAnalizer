
from fine_tune_model import model_cfg, sam2_checkpoint, test_data  # Import necessary configurations and test data  

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



def read_image(image_path, mask_path):  # read and resize image and mask
   img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
   mask = cv2.imread(mask_path, 0)
   r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
   img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
   mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
   return img, mask
 
def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   labels = [] #remove it and its mentions if  the code breaks
   
   if len(coords) == 0:
      return np.array([]), np.array([])
   
   for i in range(num_points):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([[yx[1], yx[0]]])
       labels.append(1)  # Label 1 for foreground points
   return np.array(points), np.array(labels) 

# Randomly select a test image from the test_data
selected_entry = random.choice(test_data)
print(selected_entry)
image_path = selected_entry['image']
mask_path = selected_entry['annotation']
print(mask_path,'mask path')
 
# Load the selected image and mask
image, target_mask = read_image(image_path, mask_path)
 
# Generate random points for the input
num_samples = 30  # Number of points per segment to sample
input_points, input_labels = get_points(target_mask, num_samples) #remove input_labels if the code breaks


# Add some negative points (background points) #Remove it and its mentions if the code breaks
bg_coords = np.argwhere(target_mask == 0)
if len(bg_coords) > 0:
   for i in range(10):  # Add 10 background points
       yx = np.array(bg_coords[np.random.randint(len(bg_coords))])
       input_points = np.append(input_points, [[yx[1], yx[0]]], axis=0)
       input_labels = np.append(input_labels, [0])  # Label 0 for background points


    
# Load the fine-tuned model
#TODO: add the name of the fine-tuned model
FINE_TUNED_MODEL_WEIGHTS = "./fine_tuned_models/cork_analizer_sam2.pt"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
 
# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
 
 # Check if weights file exists and load properly
if os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
    checkpoint = torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location="cuda")
    predictor.model.load_state_dict(checkpoint, strict=False)
    print("Fine-tuned weights loaded successfully")
else:
    print("Fine-tuned weights file not found!")
 
"""
# Perform inference and predict masks
with torch.no_grad():
   predictor.set_image(image)
   masks, scores, logits = predictor.predict(
       point_coords=input_points,
       point_labels=np.ones([input_points.shape[0], 1])
   )
"""
   
# Perform inference and predict masks
with torch.no_grad():
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_points.reshape(1, -1, 2),  # Proper shape
        point_labels=input_labels.reshape(1, -1)
    )
 
# Process the predicted masks and sort by scores
np_masks = np.array(masks[:, 0])
np_scores = scores[:, 0]
sorted_masks = np_masks[np.argsort(np_scores)][::-1]
 
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
 
 # Debug: Check mask quality
print(f"Mask shape: {target_mask.shape}")
print(f"Mask unique values: {np.unique(target_mask)}")
print(f"Mask non-zero pixels: {np.sum(target_mask > 0)}")
print(f"Input points shape: {input_points.shape}")
print(f"Input labels shape: {input_labels.shape}")

# Visualization: Show the original image, mask, and final segmentation side by side
plt.figure(figsize=(18, 6))
 
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



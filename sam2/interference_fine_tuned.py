import random
import cv2
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_preparation import *

def read_image(image_path, mask_path):  # read and resize image and mask
   img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
   mask = cv2.imread(mask_path, 0)
   
   # Debug: Check if mask is loaded correctly
   print(f"Loading mask from: {mask_path}")
   print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
   print(f"Mask min/max: {mask.min()}/{mask.max() if mask is not None else 'None/None'}")
   print(f"Non-zero pixels: {np.count_nonzero(mask) if mask is not None else 0}")
   
   if mask is None:
       raise ValueError(f"Could not load mask from {mask_path}")
   
   r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
   img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
   mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
   
   # Debug: Check after resize
   print(f"After resize - Mask shape: {mask.shape}")
   print(f"After resize - Non-zero pixels: {np.count_nonzero(mask)}")
   
   return img, mask

def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   
   print(f"Found {len(coords)} coordinates with mask > 0")
   
   if len(coords) == 0:
       print("WARNING: No non-zero pixels found in mask!")
       # Return a default point in the center if no mask pixels found
       center_y, center_x = mask.shape[0] // 2, mask.shape[1] // 2
       return np.array([[center_x, center_y]])  # Fixed: Remove extra nesting
   
   # Sample fewer points if not enough coordinates available
   actual_num_points = min(num_points, len(coords))
   print(f"Sampling {actual_num_points} points from {len(coords)} available coordinates")
   
   for i in range(actual_num_points):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([yx[1], yx[0]])  # Fixed: Match training format [x, y]
   
   points = np.array(points)
   points = np.expand_dims(points, axis=1)  # Fixed: Match training format (N, 1, 2)
   print(f"Final points shape: {points.shape}")
   return points

   # Randomly select a test image from the test_data
selected_entry = random.choice(test_data)
image_path = selected_entry['image']
mask_path = selected_entry['annotation']

print(f"Selected image: {image_path}")
print(f"Selected mask: {mask_path}")
print(f"Image exists: {os.path.exists(image_path)}")
print(f"Mask exists: {os.path.exists(mask_path)}")

# Load the selected image and mask
image, mask = read_image(image_path, mask_path)

# Generate random points for the input
num_samples = 30  # Number of points per segment to sample
input_points = get_points(mask, num_samples)

# Load the fine-tuned model
FINE_TUNED_MODEL_WEIGHTS = "cork_analizer_sam2_CAWR_8000.pt"  # Use the latest model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt" # Ensure this matches the training checkpoint
model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml" # Ensure this matches the training configuration

print(f"Loading model from {sam2_checkpoint}")
print(f"Loading fine-tuned weights from {FINE_TUNED_MODEL_WEIGHTS}")
print(f"Fine-tuned weights exist: {os.path.exists(FINE_TUNED_MODEL_WEIGHTS)}")

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: CUDA not available, using CPU. This will be very slow!")

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)

if os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
    # Load weights with proper device mapping
    if device == "cpu":
        weights = torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location='cpu')
    else:
        weights = torch.load(FINE_TUNED_MODEL_WEIGHTS)
    
    predictor.model.load_state_dict(weights)
    print(f"Successfully loaded fine-tuned weights")
else:
    print(f"WARNING: Fine-tuned weights not found! Using base SAM2 model.")
    print(f"Available model files:")
    for f in os.listdir('.'):
        if f.endswith('.pt'):
            print(f"  - {f}")

# Perform inference and predict masks
with torch.no_grad():
   predictor.set_image(image)
   print(f"Input points shape: {input_points.shape}")
   print(f"Input points: {input_points[:3]}")  # Show first 3 points
   
   masks, scores, logits = predictor.predict(
       point_coords=input_points,
       point_labels=np.ones([input_points.shape[0], 1])  # Fixed: Match training format
   )
   
   print(f"Predicted masks shape: {masks.shape}")
   print(f"Scores: {scores}")
   print(f"Number of masks with score > 0.5: {np.sum(scores > 0.5)}")
   
   # Debug: Check what the model actually predicted
   for i, (mask_pred, score) in enumerate(zip(masks, scores)):
       mask_pixels = np.count_nonzero(mask_pred[0])
       mask_mean = np.mean(mask_pred[0])
       mask_max = np.max(mask_pred[0])
       print(f"Mask {i}: Score={score[0]:.4f}, Non-zero pixels={mask_pixels}, Mean={mask_mean:.4f}, Max={mask_max:.4f}")
   
   # Check if the model is producing any meaningful output
   if np.all(scores < 0.1):
       print("WARNING: All prediction scores are very low! This suggests training issues.")
   
   if all(np.count_nonzero(mask[0]) < 100 for mask in masks):
       print("WARNING: All predicted masks have very few pixels! This suggests training issues.")

# Process the predicted masks and sort by scores
if len(masks) > 0:
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]
    
    print(f"Processing {len(sorted_masks)} predicted masks")
    print(f"Best score: {np_scores.max():.4f}")
    
    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask_pred = sorted_masks[i]
        
        # Skip if overlap is too high
        if mask_pred.sum() > 4:  # Only check if mask has content #0
            overlap_ratio = (mask_pred * occupancy_mask).sum() / mask_pred.sum()
            if overlap_ratio > 4.15: #0.15:
                print(f"Skipping mask {i} due to high overlap ({overlap_ratio:.3f})")
                continue
        
        mask_bool = mask_pred.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
        seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
        occupancy_mask[mask_bool] = True  # Update occupancy_mask
        
        print(f"Added mask {i} with {mask_bool.sum()} pixels")
else:
    print("No masks predicted!")
    seg_map = np.zeros_like(mask, dtype=np.uint8)

# Visualization: Show the original image, mask, and final segmentation side by side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Test Image - ' + os.path.basename(image_path))
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Original Mask - ' + os.path.basename(mask_path))
# Force proper contrast for mask display
mask_display = mask.copy()
if mask_display.max() > 0:
    mask_display = (mask_display / mask_display.max() * 255).astype(np.uint8)
plt.imshow(mask_display, cmap='gray', vmin=0, vmax=255)
plt.colorbar(label='Mask values')
plt.axis('off')

# Add text showing mask statistics
plt.text(0.02, 0.98, f'Non-zero pixels: {np.count_nonzero(mask)}\nMax value: {mask.max()}\nMin value: {mask.min()}', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.subplot(1, 3, 3)
plt.title('Final Segmentation')
plt.imshow(seg_map, cmap='jet')
plt.colorbar(label='Segment ID')
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional debug visualization: Show individual predicted masks
if len(masks) > 0:
    num_masks_to_show = min(3, len(masks))
    plt.figure(figsize=(15, 5))
    
    for i in range(num_masks_to_show):
        plt.subplot(1, num_masks_to_show, i + 1)
        plt.title(f'Predicted Mask {i+1}\nScore: {scores[i, 0]:.4f}')
        plt.imshow(masks[i, 0], cmap='gray')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show overlay of best mask on original image
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth Overlay')
    plt.imshow(image)
    # Fix: Ensure mask is properly binarized for visualization
    mask_binary = (mask > 0).astype(np.float32)
    plt.imshow(mask_binary, alpha=0.4, cmap='Reds')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Best Prediction Overlay')
    plt.imshow(image)
    # Fix 1: Use the mask with highest score for overlay
    best_mask_idx = np.argmax(scores[:, 0])
    best_mask = masks[best_mask_idx, 0]
    # Fix 2: Threshold the predicted mask for better visualization
    best_mask_binary = (best_mask > 0.5).astype(np.float32)
    plt.imshow(best_mask_binary, alpha=0.4, cmap='Blues')
    plt.text(10, 20, f"Score: {scores[best_mask_idx, 0]:.4f}", 
             color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    plt.axis('off')
    
    # Add IoU calculation between ground truth and best prediction
    intersection = np.logical_and(mask_binary, best_mask_binary).sum()
    union = np.logical_or(mask_binary, best_mask_binary).sum()
    iou = intersection / union if union > 0 else 0
    plt.suptitle(f'Ground Truth vs Best Prediction (IoU: {iou:.4f})', fontsize=14)
    
    # Add another visualization showing all top 3 predictions combined
    if len(masks) >= 3:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title('Ground Truth')
        plt.imshow(image)
        plt.imshow(mask_binary, alpha=0.4, cmap='Reds')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Best Single Prediction')
        plt.imshow(image)
        plt.imshow(best_mask_binary, alpha=0.4, cmap='Blues')
        plt.axis('off')
        
        # Show top 3 predictions combined
        plt.subplot(1, 3, 3)
        plt.title('Top 3 Predictions Combined')
        plt.imshow(image)
        
        # Create a combined mask from top 3 predictions
        combined_mask = np.zeros_like(best_mask)
        top_indices = np.argsort(scores[:, 0])[-3:][::-1]  # Get indices of top 3 scores
        
        # Use different colors for each prediction
        colors = ['Blues', 'Greens', 'Purples']
        for i, idx in enumerate(top_indices):
            pred_mask = (masks[idx, 0] > 0.5).astype(np.float32)
            # Use different alpha and cmap for each mask
            plt.imshow(pred_mask, alpha=0.3, cmap=colors[i])
            combined_mask = np.logical_or(combined_mask, pred_mask)
        
        # Calculate IoU for combined mask
        combined_iou = np.logical_and(mask_binary, combined_mask).sum() / np.logical_or(mask_binary, combined_mask).sum()
        plt.title(f'Top 3 Predictions Combined\nIoU: {combined_iou:.4f}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
else:
    print("No masks to visualize")

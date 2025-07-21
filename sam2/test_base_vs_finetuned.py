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

def read_image(image_path, mask_path):
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask, num_points):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        center_y, center_x = mask.shape[0] // 2, mask.shape[1] // 2
        return np.array([[[center_x, center_y]]])
    
    points = []
    actual_num_points = min(num_points, len(coords))
    for i in range(actual_num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

# Test with a specific image
image_path = "./dataset/images/866b4db3-rolha000020.png"
mask_path = "./dataset/masks/866b4db3-rolha000020_mask.png"

print("Testing BASE SAM2 model (without fine-tuning)")
print(f"Image: {image_path}")
print(f"Mask: {mask_path}")

# Load image and mask
image, mask = read_image(image_path, mask_path)
print(f"Loaded - Image shape: {image.shape}, Mask shape: {mask.shape}")
print(f"Mask non-zero pixels: {np.count_nonzero(mask)}")

# Get points from ground truth
input_points = get_points(mask, 10)
print(f"Input points: {input_points.shape}")

# Load BASE SAM2 model (no fine-tuning)
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

print("Running inference with BASE SAM2 model...")

# Test BASE model first
with torch.no_grad():
    predictor.set_image(image)
    masks_base, scores_base, logits_base = predictor.predict(
        point_coords=input_points,
        point_labels=np.ones([input_points.shape[0], 1])
    )

print(f"BASE model results:")
print(f"Masks shape: {masks_base.shape}")
print(f"Scores: {scores_base}")

for i, (mask_pred, score) in enumerate(zip(masks_base, scores_base)):
    mask_pixels = np.count_nonzero(mask_pred[0])
    print(f"BASE Mask {i}: Score={score[0]:.4f}, Non-zero pixels={mask_pixels}")

# Now test FINE-TUNED model
FINE_TUNED_MODEL_WEIGHTS = "cork_analizer_sam2_8000.pt"

if os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
    print(f"\nLoading fine-tuned weights: {FINE_TUNED_MODEL_WEIGHTS}")
    
    if device == "cpu":
        weights = torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location='cpu')
    else:
        weights = torch.load(FINE_TUNED_MODEL_WEIGHTS)
    
    predictor.model.load_state_dict(weights)
    print("Fine-tuned weights loaded successfully")
    
    print("Running inference with FINE-TUNED model...")
    
    with torch.no_grad():
        predictor.set_image(image)
        masks_ft, scores_ft, logits_ft = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    print(f"FINE-TUNED model results:")
    print(f"Masks shape: {masks_ft.shape}")
    print(f"Scores: {scores_ft}")
    
    for i, (mask_pred, score) in enumerate(zip(masks_ft, scores_ft)):
        mask_pixels = np.count_nonzero(mask_pred[0])
        print(f"FINE-TUNED Mask {i}: Score={score[0]:.4f}, Non-zero pixels={mask_pixels}")
    
    # Compare results
    plt.figure(figsize=(20, 10))
    
    # Ground truth
    plt.subplot(2, 4, 1)
    plt.title('Ground Truth Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.title('Ground Truth Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # BASE model results
    plt.subplot(2, 4, 2)
    plt.title('BASE SAM2 - Best Mask')
    if len(masks_base) > 0:
        best_base = masks_base[np.argmax(scores_base[:, 0]), 0]
        plt.imshow(best_base, cmap='gray')
        plt.text(0.02, 0.98, f'Score: {scores_base.max():.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.title('BASE SAM2 - Overlay')
    plt.imshow(image)
    if len(masks_base) > 0:
        best_base = masks_base[np.argmax(scores_base[:, 0]), 0]
        plt.imshow(best_base, alpha=0.5, cmap='Blues')
    plt.axis('off')
    
    # Fine-tuned model results
    plt.subplot(2, 4, 3)
    plt.title('Fine-tuned - Best Mask')
    if len(masks_ft) > 0:
        best_ft = masks_ft[np.argmax(scores_ft[:, 0]), 0]
        plt.imshow(best_ft, cmap='gray')
        plt.text(0.02, 0.98, f'Score: {scores_ft.max():.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.title('Fine-tuned - Overlay')
    plt.imshow(image)
    if len(masks_ft) > 0:
        best_ft = masks_ft[np.argmax(scores_ft[:, 0]), 0]
        plt.imshow(best_ft, alpha=0.5, cmap='Reds')
    plt.axis('off')
    
    # Comparison
    plt.subplot(2, 4, 4)
    plt.title('Score Comparison')
    if len(masks_base) > 0 and len(masks_ft) > 0:
        plt.bar(['BASE', 'Fine-tuned'], [scores_base.max(), scores_ft.max()])
        plt.ylabel('Best Score')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.title('Points on Image')
    plt.imshow(image)
    for point in input_points:
        plt.scatter(point[0][0], point[0][1], c='red', s=50, marker='x')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

else:
    print(f"Fine-tuned weights not found: {FINE_TUNED_MODEL_WEIGHTS}")

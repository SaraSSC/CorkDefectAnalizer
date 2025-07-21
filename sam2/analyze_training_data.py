import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import *

def analyze_training_data():
    """Analyze the quality of training data"""
    print("Analyzing training data quality...")
    
    # Sample some training examples
    mask_sizes = []
    mask_qualities = []
    
    for i in range(min(10, len(train_data))):
        entry = train_data[i]
        
        # Load mask
        mask = cv2.imread(entry['annotation'], 0)
        image = cv2.imread(entry['image'])[..., ::-1]
        
        if mask is None or image is None:
            continue
            
        # Resize to training size
        r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
        mask_resized = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        
        # Analyze mask
        non_zero_pixels = np.count_nonzero(mask_resized)
        total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
        mask_ratio = non_zero_pixels / total_pixels
        
        mask_sizes.append(non_zero_pixels)
        mask_qualities.append(mask_ratio)
        
        print(f"Sample {i+1}: {os.path.basename(entry['image'])}")
        print(f"  - Non-zero pixels: {non_zero_pixels}")
        print(f"  - Mask ratio: {mask_ratio:.4f}")
        print(f"  - Mask size: {mask_resized.shape}")
        
        # Check if mask is too small or too large
        if mask_ratio < 0.001:
            print(f"  WARNING: Very small mask (< 0.1% of image)")
        elif mask_ratio > 0.5:
            print(f"  WARNING: Very large mask (> 50% of image)")
    
    # Statistics
    print(f"\nTraining Data Statistics:")
    print(f"Average mask size: {np.mean(mask_sizes):.0f} pixels")
    print(f"Average mask ratio: {np.mean(mask_qualities):.4f}")
    print(f"Mask size range: {np.min(mask_sizes):.0f} - {np.max(mask_sizes):.0f}")
    
    # Visualize some examples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(min(4, len(train_data))):
        entry = train_data[i]
        
        image = cv2.imread(entry['image'])[..., ::-1]
        mask = cv2.imread(entry['annotation'], 0)
        
        # Resize
        r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
        image_resized = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
        mask_resized = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        
        # Plot
        axes[0, i].imshow(image_resized)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_resized, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}\n{np.count_nonzero(mask_resized)} pixels')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_point_sampling():
    """Check if point sampling is working correctly"""
    print("\nTesting point sampling...")
    
    # Test with a few samples
    for i in range(3):
        image, mask, points, num_masks = read_batch(train_data, visualize_data=False)
        
        if image is None or mask is None:
            continue
            
        print(f"Sample {i+1}:")
        print(f"  - Image shape: {image.shape}")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Points shape: {points.shape}")
        print(f"  - Number of masks: {num_masks}")
        print(f"  - Points: {points}")
        
        # Check if points are inside mask
        if mask.ndim > 2:
            mask_2d = mask[0]
        else:
            mask_2d = mask
            
        valid_points = 0
        for point in points:
            x, y = int(point[0][0]), int(point[0][1])
            if 0 <= x < mask_2d.shape[1] and 0 <= y < mask_2d.shape[0]:
                if mask_2d[y, x] > 0:
                    valid_points += 1
        
        print(f"  - Valid points inside mask: {valid_points}/{len(points)}")
        
        if valid_points == 0:
            print("  ERROR: No valid points found inside mask!")

if __name__ == "__main__":
    analyze_training_data()
    check_point_sampling()

"""
Cork Defect Prediction and Visualization Script
This script uses a fine-tuned SAM2 model to predict defects in cork images
without requiring existing masks. It provides visualization with defect analysis
including mask overlay, defect numbering, and a detailed results table.

Author: SaraSSC
Date: July 2025
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch
from scipy import ndimage
from skimage import measure
import random

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class CorkDefectPredictor:
    def __init__(self, model_cfg, sam2_checkpoint, fine_tuned_weights=None):
        """
        Initialize the Cork Defect Predictor
        
        Args:
            model_cfg: Path to model configuration file
            sam2_checkpoint: Path to base SAM2 checkpoint
            fine_tuned_weights: Path to fine-tuned weights (optional)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Load fine-tuned weights if provided
        if fine_tuned_weights and os.path.exists(fine_tuned_weights):
            if self.device == "cpu":
                weights = torch.load(fine_tuned_weights, map_location='cpu')
            else:
                weights = torch.load(fine_tuned_weights)
            self.predictor.model.load_state_dict(weights)
            print(f"Successfully loaded fine-tuned weights from {fine_tuned_weights}")
        else:
            print("Using base SAM2 model (no fine-tuned weights loaded)")
    
    def preprocess_image(self, image_path, target_size=1024):
        """
        Load and preprocess image for inference
        
        Args:
            image_path: Path to input image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image as RGB numpy array
        """
        img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        
        # Resize maintaining aspect ratio
        r = np.min([target_size / img.shape[1], target_size / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        
        return img
    
    def generate_grid_points(self, image_shape, grid_density=32, add_random=True, random_points=20):
        """
        Generate a grid of points for automatic mask generation
        with optional random points for better defect coverage
        
        Args:
            image_shape: Shape of the image (H, W)
            grid_density: Number of points per dimension for the grid
            add_random: Whether to add random points in addition to the grid
            random_points: Number of random points to add
            
        Returns:
            Grid points as numpy array with proper shape for SAM2
        """
        h, w = image_shape[:2]
        
        # Use adaptive grid density based on image size
        # This prevents excessive points for large images
        area = h * w
        adaptive_density = min(grid_density, max(16, int(np.sqrt(area) // 50)))
        print(f"Using adaptive grid density: {adaptive_density}")
        
        # Create grid with wider spacing for faster processing
        # Focus on covering key areas rather than a dense uniform grid
        x_coords = np.linspace(w//16, w-w//16, adaptive_density)
        y_coords = np.linspace(h//16, h-h//16, adaptive_density)
        
        points = []
        # Use strided sampling to reduce total point count
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    points.append([x, y])
        
        # Add focused random points in limited quantity
        if add_random:
            # Add some random points but limit to a reasonable number
            random_points = min(random_points, 30)
            
            for _ in range(random_points):
                x = np.random.randint(w//20, w-w//20)
                y = np.random.randint(h//20, h-h//20)
                points.append([x, y])
            
            # Add a few edge-focused points (reduced from 20)
            edge_points = 12
            for _ in range(edge_points):
                edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    x = np.random.randint(w//10, w-w//10)
                    y = np.random.randint(h//20, h//5)
                elif edge == 'bottom':
                    x = np.random.randint(w//10, w-w//10)
                    y = np.random.randint(h-h//5, h-h//20)
                elif edge == 'left':
                    x = np.random.randint(w//20, w//5)
                    y = np.random.randint(h//10, h-h//10)
                else:  # right
                    x = np.random.randint(w-w//5, w-w//20)
                    y = np.random.randint(h//10, h-h//10)
                points.append([x, y])
                
            # Use fewer cluster centers but maintain precision
            cluster_centers = 5
            points_per_cluster = 8
            
            for _ in range(cluster_centers):
                cx = np.random.randint(w//10, w-w//10)
                cy = np.random.randint(h//10, h-h//10)
                
                for _ in range(points_per_cluster):
                    offset_x = int(np.random.normal(0, w//40))
                    offset_y = int(np.random.normal(0, h//40))
                    x = max(0, min(w-1, cx + offset_x))
                    y = max(0, min(h-1, cy + offset_y))
                    points.append([x, y])
        
        # Convert to numpy array and limit total number of points for performance
        points = points[:min(len(points), 300)]  # Cap total points at 300
        points_array = np.array(points)
        points_array = np.expand_dims(points_array, axis=1)
        
        print(f"Generated {len(points)} points for prediction")
        return points_array
    
    def predict_defects(self, image, num_grid_points=32, score_threshold=0.2):
        """
        Predict defects in the given image using grid points
        
        Args:
            image: Input image as numpy array
            num_grid_points: Number of grid points for prediction
            score_threshold: Minimum confidence score for defects
            
        Returns:
            Tuple of (masks, scores, processing_time)
        """
        start_time = time.time()
        
        # Set image for predictor
        self.predictor.set_image(image)
        
        # Generate grid points plus random points for better detection
        grid_points = self.generate_grid_points(image.shape, 
                                              grid_density=num_grid_points,
                                              add_random=True,
                                              random_points=30)
        
        # Early termination check
        if grid_points.shape[0] == 0:
            print("No valid points generated for prediction")
            return np.array([]), np.array([]), time.time() - start_time
        
        # Prepare point labels (all foreground)
        point_labels = np.ones((grid_points.shape[0], 1), dtype=np.int64)
        
        try:
            # Process points in batches to save memory and improve performance
            batch_size = 100 #50
            all_masks = []
            all_scores = []
            
            # Process in batches
            for i in range(0, len(grid_points), batch_size):
                batch_points = grid_points[i:i+batch_size]
                batch_labels = point_labels[i:i+batch_size]
                
                # Track batch time for diagnostics
                batch_start = time.time()
                
                with torch.no_grad():
                    masks, scores, _ = self.predictor.predict(
                        point_coords=batch_points,
                        point_labels=batch_labels,
                        multimask_output=False, #True
                    )
                
                # Filter low-quality masks early
                if len(scores.shape) == 1:
                    valid_indices = scores > score_threshold
                else:
                    valid_indices = scores[:, 0] > score_threshold
                
                if np.any(valid_indices):
                    all_masks.append(masks[valid_indices])
                    all_scores.append(scores[valid_indices])
                
                batch_end = time.time()
                print(f"Processed batch {i//batch_size + 1}/{(len(grid_points) + batch_size - 1)//batch_size} in {batch_end-batch_start:.2f}s")
            
            # Combine results from batches
            if all_masks and all_scores:
                valid_masks = np.vstack(all_masks) if len(all_masks) > 0 else np.array([])
                valid_scores = np.vstack(all_scores) if len(all_scores) > 0 else np.array([])
            else:
                valid_masks = np.array([])
                valid_scores = np.array([])
            
            processing_time = time.time() - start_time
            
            print(f"Found {len(valid_masks)} valid masks after filtering")
            print(f"Total processing time: {processing_time:.2f}s")
            
            return valid_masks, valid_scores, processing_time
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Grid points shape: {grid_points.shape}")
            print(f"Point labels shape: {point_labels.shape}")
            
            # Return empty results
            processing_time = time.time() - start_time
            return np.array([]), np.array([]), processing_time
    
    def process_masks(self, masks, scores, overlap_threshold=0.15):
        """
        Process and combine overlapping masks
        
        Args:
            masks: Predicted masks
            scores: Confidence scores
            overlap_threshold: Maximum allowed overlap ratio
            
        Returns:
            Processed segmentation map and defect info
        """
        if len(masks) == 0:
            return np.zeros((256, 256), dtype=np.uint8), []
        
        # Sort masks by confidence score (highest first)
        # Handle both 1D and 2D score arrays
        if len(scores.shape) == 1:
            score_values = scores
        else:
            score_values = scores[:, 0]
            
        sorted_indices = np.argsort(score_values)[::-1]
        sorted_masks = masks[sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Initialize segmentation map
        seg_map = np.zeros_like(sorted_masks[0, 0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0, 0], dtype=bool)
        
        defect_info = []
        defect_id = 1
        
        for i, (mask, score) in enumerate(zip(sorted_masks, sorted_scores)):
            # Handle different mask formats
            if mask.ndim == 3:
                mask_binary = mask[0].astype(bool)
            else:
                mask_binary = mask.astype(bool)
                
            # Extract score value (handle both 1D and 2D arrays)
            if isinstance(score, np.ndarray) and len(score.shape) > 0 and score.shape[0] > 0:
                score_value = score[0] if len(score.shape) > 0 else score
            else:
                score_value = float(score)
            
            # Skip if mask is too small (lowered minimum size for better defect detection)
            if np.sum(mask_binary) < 50:  # Minimum 50 pixels (was 100)
                continue
            
            # Check overlap with existing defects - allow more overlap
            if np.sum(mask_binary) > 0:
                overlap_ratio = np.sum(mask_binary & occupancy_mask) / np.sum(mask_binary)
                if overlap_ratio > overlap_threshold:
                    continue
            
            # Add mask to segmentation map
            non_overlapping_mask = mask_binary & ~occupancy_mask
            seg_map[non_overlapping_mask] = defect_id
            occupancy_mask[non_overlapping_mask] = True
            
            # Calculate defect properties
            defect_props = self.calculate_defect_properties(non_overlapping_mask)
            defect_info.append({
                'id': defect_id,
                'confidence': score_value,
                'area': defect_props['area'],
                'width': defect_props['width'],
                'height': defect_props['height'],
                'centroid': defect_props['centroid']
            })
            
            defect_id += 1
        
        return seg_map, defect_info
    
    def calculate_defect_properties(self, mask):
        """
        Calculate properties of a defect mask
        
        Args:
            mask: Binary mask of the defect
            
        Returns:
            Dictionary with defect properties
        """
        # Find contours
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) == 0:
            return {'area': 0, 'width': 0, 'height': 0, 'centroid': (0, 0)}
        
        # Get largest region
        largest_region = max(regions, key=lambda r: r.area)
        
        # Calculate bounding box
        min_row, min_col, max_row, max_col = largest_region.bbox
        width = max_col - min_col
        height = max_row - min_row
        
        return {
            'area': largest_region.area,
            'width': width,
            'height': height,
            'centroid': largest_region.centroid
        }

def create_visualization(image, seg_map, defect_info, image_name, processing_time):
    """
    Create visualization of cork defect detection results
    
    Args:
        image: Original image
        seg_map: Segmentation map with defect IDs
        defect_info: List of defect information dictionaries
        image_name: Name of the input image
        processing_time: Time taken for processing
    """
    # Create figure with stacked layout - image on top, table below
    fig = plt.figure(figsize=(14, 10))
    
    # Configure grid layout with 2 rows
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Title with image name, processing time, and defect count
    num_defects = len(defect_info)
    title = f"{image_name} | Processing Time: {processing_time:.2f}s | Defects Found: {num_defects}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Image panel (contains original image and overlay side by side)
    image_panel = fig.add_subplot(gs[0])
    
    # Display original image
    image_panel.imshow(image)
    image_panel.set_title('Cork Image with Defect Detection', fontsize=14)
    image_panel.axis('off')
    
    # Create colors for defects
    if num_defects > 0:
        # Use distinct colors for different defects
        colors = plt.cm.tab10(np.linspace(0, 1, max(num_defects, 10)))
        
        # Overlay segmentation map with transparency
        # First, create a colored overlay image
        h, w = seg_map.shape
        overlay_img = np.zeros((h, w, 4))  # RGBA
        
        for i, defect in enumerate(defect_info):
            defect_id = defect['id']
            mask = (seg_map == defect_id)
            
            # Set RGB values from our color palette
            overlay_img[mask, 0] = colors[i % len(colors)][0]  # R
            overlay_img[mask, 1] = colors[i % len(colors)][1]  # G
            overlay_img[mask, 2] = colors[i % len(colors)][2]  # B
            overlay_img[mask, 3] = 0.6  # Alpha (transparency)
        
        # Display the overlay
        image_panel.imshow(overlay_img, alpha=0.6)
        
        # Add defect numbers at centroid of each defect
        for defect in defect_info:
            centroid = defect['centroid']
            # Add defect number with white background for visibility
            image_panel.annotate(str(defect['id']), 
                        xy=(centroid[1], centroid[0]),
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                        fontsize=12, fontweight='bold', color='black')
            
            # Add bounding box around defect
            mask_coords = np.where(seg_map == defect['id'])
            if len(mask_coords[0]) > 0:
                min_row, max_row = mask_coords[0].min(), mask_coords[0].max()
                min_col, max_col = mask_coords[1].min(), mask_coords[1].max()
                
                rect = patches.Rectangle((min_col, min_row), 
                                       max_col - min_col, max_row - min_row,
                                       linewidth=2, edgecolor='red', facecolor='none')
                image_panel.add_patch(rect)
    
    # Table panel for defect information
    table_panel = fig.add_subplot(gs[1])
    table_panel.axis('off')
    
    if num_defects > 0:
        # Prepare data for the table - exactly 3 columns as requested
        table_data = []
        headers = ['Defect #', 'Dimensions (W×H px)', 'Confidence (%)']
        
        for defect in defect_info:
            row = [
                str(defect['id']),  # Defect number
                f"{defect['width']}×{defect['height']}",  # Dimensions
                f"{defect['confidence']*100:.1f}%"  # Confidence
            ]
            table_data.append(row)
        
        # Create table
        table = table_panel.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0.05, 0.1, 0.9, 0.8])  # Adjust table position and size
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.8)  # Make rows taller
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')  # Green header
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors matching the defect colors
        for i, defect in enumerate(defect_info):
            # Use same colors as in the overlay
            color = colors[i % len(colors)]
            lighter_color = color * 0.5 + 0.5  # Make it lighter
            
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(lighter_color)
    else:
        # No defects found message
        table_panel.text(0.5, 0.5, 'No defects detected in this image', 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Make room for title
    return fig

def main():
    """
    Main function to run defect detection and visualization on cork images
    """
    # Configuration - ADJUST THESE PATHS AS NEEDED
    MODEL_CFG = "./configs/sam2.1/sam2.1_hiera_b+.yaml"  # Model configuration
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"  # Base SAM2 checkpoint
    
    # Try to find the fine-tuned weights in different possible locations
    possible_weight_paths = [
        "./cork_analizer_sam2_CAWR_8000.pt",
          # Try without CAWR suffix
        
        *[f for f in os.listdir(".") if f.endswith('.pt') and 'cork' in f.lower()]
    ]
    
    FINE_TUNED_WEIGHTS = None
    for path in possible_weight_paths:
        if os.path.exists(path):
            FINE_TUNED_WEIGHTS = path
            print(f"Found fine-tuned weights: {path}")
            break
            
    if FINE_TUNED_WEIGHTS is None:
        print("No fine-tuned weights found. Using base SAM2 checkpoint only.")
    
    # Process a single image or all images in a folder
    PROCESS_ALL = True  # Set to True to process all images in the folder
    
    # Create folders for test images and results
    test_images_dir = "./test_images"
    output_dir = "./defect_analysis_results"
    for directory in [test_images_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Initialize predictor
    print("Initializing Cork Defect Predictor...")
    predictor = CorkDefectPredictor(MODEL_CFG, SAM2_CHECKPOINT, FINE_TUNED_WEIGHTS)
    
    # Select image source
    if os.path.exists(test_images_dir) and len([f for f in os.listdir(test_images_dir) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]) > 0:
        # Use images from test_images folder
        test_files = [f for f in os.listdir(test_images_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_dir = test_images_dir
        print(f"Using images from test_images folder: {len(test_files)} images found")
    else:
        # Use sample images from dataset
        image_dir = "./dataset/images"
        test_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Using sample images from dataset: {len(test_files)} images found")
    
    if len(test_files) == 0:
        print("No images found to process!")
        return
        
    # Process either a single random image or all images
    if not PROCESS_ALL:
        test_files = [random.choice(test_files)]
        print(f"Processing a single random image: {test_files[0]}")
    
    # Process each selected image
    for image_file in test_files:
        image_path = os.path.join(image_dir, image_file)
        image_name = os.path.basename(image_path)
        
        print(f"\nProcessing image: {image_name}")
        
        # Load and preprocess image
        image = predictor.preprocess_image(image_path)
        print(f"Image shape: {image.shape}")
        
        # Add performance metrics
        print(f"Memory usage before prediction: {torch.cuda.memory_allocated() / 1e9:.2f} GB") if torch.cuda.is_available() else None
        
        # Start with a reasonable grid density for speed
        masks, scores, processing_time = predictor.predict_defects(
            image, 
            num_grid_points=10, #32  # Lower initial grid density for faster processing
            score_threshold=0.08   # Slightly lower threshold for better sensitivity
        )
        
        print(f"Initial predictions: {len(masks)} masks")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # If no defects found with fast settings, try more intensive settings once
        if len(masks) == 0:
            print("No defects found with fast settings, trying more intensive search...")
            masks, scores, processing_time = predictor.predict_defects(
                image, 
                num_grid_points=20, #48
                score_threshold=0.06#0.05
            )
            print(f"Secondary predictions: {len(masks)} masks")
            print(f"Total processing time: {processing_time:.2f} seconds")
        
        # Process masks and extract defect information with adaptive threshold
        overlap_threshold = 0.25 if len(masks) < 10 else 0.4  # Use different threshold based on mask count
        seg_map, defect_info = predictor.process_masks(
            masks, 
            scores,
            overlap_threshold=overlap_threshold
        )
        
        print(f"Final defects after processing: {len(defect_info)}")
        
        # Create visualization
        fig = create_visualization(image, seg_map, defect_info, image_name, processing_time)
        
        # Save results
        output_path = os.path.join(output_dir, f"analysis_{image_name.split('.')[0]}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        
        # Show visualization
        #plt.show()
        
        # Print detailed results
        print("\n" + "="*60)
        print(f"DEFECT ANALYSIS SUMMARY - {image_name}")
        print("="*60)
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Total Defects Found: {len(defect_info)}")
        
        if defect_info:
            print("\nDefect Details:")
            for defect in defect_info:
                print(f"  Defect #{defect['id']}:")
                print(f"    - Confidence: {defect['confidence']*100:.1f}%")
                print(f"    - Dimensions: {defect['width']}×{defect['height']} pixels")
                print(f"    - Area: {defect['area']} pixels")
        
        print("="*60)
        print("To analyze more images, add them to the './test_images' folder and run again.")

if __name__ == "__main__":
    main()

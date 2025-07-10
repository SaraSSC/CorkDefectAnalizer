import os
import random
import cv2
import torch
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

"""
Setting the seed for reproducibility.
    For deterministic results and reproducibility, we will set a fixed seed value to ensure consistent runs across different runs.
    This is a very common strategy for Finetuning SAM2 or any other model.

"""

def set_seeds():
    
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    print(f"Seeds set to {SEED_VALUE} for reproducibility")
set_seeds()

"""
Data loading and splitting.

 Start by defining file paths to our dataset directories. 
 The CSV file, train.csv, holds metadata pairing images (`image_path`) with their masks (`mask_path`). 
 Use train_test_split from scikit-learn to partition our data into training and testing sets, allocating 80% to training 
 and 20% to validating. Each entry in train_data and test_data is a dictionary containing the file paths for the corresponding
 image and mask, enabling easy iteration during training and validation.
"""
data_dir = "./dataset"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_data = []
for index, row in train_df.iterrows():
    image_name = row['image_id']
    mask_name = row['mask_id']
    train_data.append({
        "image" : os.path.join(images_dir, image_name),
        "annotation" : os.path.join(masks_dir, mask_name)
    })
    
test_data = []
for index, row in test_df.iterrows():
    image_name = row['image_id']
    mask_name = row['mask_id']
    test_data.append({
        "image" : os.path.join(images_dir, image_name),
        "annotation" : os.path.join(masks_dir, mask_name)
    })

"""
Data processing and visualization.

This function takes a random sample from our dataset, loads and resizes the image and mask into (1024 x 1024) as SAM2 expects
this default size for training, and consolidates the mask into a single binary representation. We are not applying any augmentations
on the data as SAM2 is capable enough to handle small dataset.

Then generates some random points on the ROI regions of the mask, which we will use as an input to the model. 
Apply light erosion on the mask to prevent sampling prompt points on boundary regions, which can sometimes confuse the model. 
This ensures each distinct diseased region is represented by at least one prompt

Finally, rearrange the mask into the shape (1, H, W) and the points into the shape (num_points, 1, 2), preparing them for input into 
the SAM2 model. 
This will be our structure of the training batch [input image, mask, the points, and the number of seg masks] for finetuning SAM2, and
this is the finest approach to train SAM2 very quickly, with less computational expenses. 

"""


#Takes a random sample from the dataset
def read_batch(data, visualize_data=True):
    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    
    if Img is None or ann_map is None:
        print(f"Error reading image or annotation for {ent['image']}")
        return None, None, None, 0
    
    #resize the image and mask to 1024x1024
    r = np.min([1024 / Img.shape[0], 1024 / Img.shape[1]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                        interpolation=cv2.INTER_NEAREST)
    
    # Consolidate the mask into a single binary representation
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []
    inds = np.unique(ann_map)[1:]
    
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask)
    
    # Generate random points on the ROI regions of the mask
    # Apply light erosion to prevent sampling prompt points on boundary regions
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded_mask > 0)
    
    if len(coords) > 0:
        for _ in inds:
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])
    points = np.array(points)
    
    if visualize_data:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(Img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Binary Mask with Points')
        plt.imshow(binary_mask, cmap='gray')
       
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    binary_mask = np.expand_dims(binary_mask, axis=1)
    binary_mask = binary_mask.transpose((2, 0, 1))  # Change mask shape into the shape (1, H, W)
    
    points = np.expand_dims(points, axis=1)  # Points into the shape above (num_points, 1, 2)
    return Img, binary_mask, points, len(inds)

#structure of the training batch [input image, mask, the points, and the number of seg masks]
Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)

def load_dataset():
    """
    Load and return the training and test datasets.
    Returns:
        tuple: (train_data, test_data) where each is a list of dictionaries
               containing image and annotation paths
    """
    return train_data, test_data

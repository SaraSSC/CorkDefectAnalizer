import os
import random
import cv2
import torch
import torch.nn.utils as utils
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
import sam2

from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

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

train_df = pd.read_csv(os.path.join(data_dir," train.csv"))

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

Then we generate some random points on the ROI regions of the mask, which we will use as an input to the model. 
Apply light erosion on the mask to prevent sampling prompt points on boundary regions, which can sometimes confuse the model. 
This ensures each distinct diseased region is represented by at least one prompt

Finally, rearrange the mask into the shape (1, H, W) and the points into the shape (num_points, 1, 2), preparing them for input into 
the SAM2 model. 
This will be our structure of the training batch [input image, mask, the points, and the number of seg masks] for finetuning SAM2, and
this is the finest approach to train SAM2 very quickly, with less computational expenses. 

"""

def read_batch(data, visualize_data = True):
    
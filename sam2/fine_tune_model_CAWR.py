from data_preparation import * 

import os
import random
import cv2
import torch
import torch.amp
#from torch.amp.autocast_mode import autocast
from torch.amp import autocast
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



# Specify the path to the SAM2 model checkpoint and configuration file
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"  # NOTE remove /sam2 path if breaks


# Check if CUDA is available and compatible, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# By initializing build_sam2 with these paths the core SAM2 model is instantiated
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# Class SAM2ImagePredictor is used to handle prompts and predictions
predictor = SAM2ImagePredictor(sam2_model)

# Training mode ensures that relevant layers can be fine-tuned when optimization routine starts
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)


scaler = GradScaler("cuda") if torch.cuda.is_available() else GradScaler("cpu")  # For mixed precision training 


NO_OF_STEPS = 8000  # Number of training steps


    

FINE_TUNED_MODEL_NAME = "cork_analizer_sam2_CAWR_small"  # Base name for the fine-tuned model


optimizer = torch.optim.AdamW(params=predictor.model.parameters(),
                              lr=0.00001,  # Current 1e-5
                              weight_decay=1e-4)

# There are different learning rate schedulers available, here I'm using CosineAnnealingWarmRestarts

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2000,
        T_mult=2,
        eta_min=1e-7,
        last_epoch=-1
    )

accumulation_steps = 8  # Gradient accumulation steps
dynamic_accumulation = True  # Enable dynamic accumulation

def get_bounding_box(ground_truth_map):
    """Get bounding box from mask with perturbation"""
    if ground_truth_map.ndim > 2:
        ground_truth_map = ground_truth_map[0]
        
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

# Training function

def train (predictor, train_data, step, mean_iou):
    with torch.amp.autocast(device_type='cuda'):
        image, mask, input_point, num_masks = read_batch(train_data, 
                                                         visualize_data=False)
        
        if image is None or mask is None or num_masks==0:
            print(f"Step {step}: Training - Early return: image={image is not None}, mask={mask is not None}, num_masks={num_masks}")
            return mean_iou
        
        input_label = np.ones((num_masks,1))
        
         # Add bounding box prompt
        bbox = get_bounding_box(mask)
        input_box = np.array([bbox]) if bbox is not None else None
        
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            print(f"Step {step}: Training - Early return: input_point type={type(input_point)}, input_label type={type(input_label)}")
            return mean_iou

        if input_point.size == 0 or input_label.size == 0:
            print(f"Step {step}: Training - Early return: input_point size={input_point.size}, input_label size={input_label.size}")
            return mean_iou

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point,
            input_label,
            box=input_box, # Use input_box for bounding box prompt
            mask_logits=None,
            normalize_coords=True
        )

        
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            print(f"Step {step}: Training - Early return: unnorm_coords={unnorm_coords is not None}, labels={labels is not None}, unnorm_box={unnorm_box is not None}")
            return mean_iou
        
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )
        
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        
        
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
         
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
         
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()
         
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
 
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.1  # Increased score loss weight
        
        # Implement dynamic accumulation properly
        effective_accum_steps = accumulation_steps
        if dynamic_accumulation and seg_loss > 1.5:  # Threshold for "bad" batches
            effective_accum_steps = accumulation_steps * 2
            
        loss = loss / effective_accum_steps
        scaler.scale(loss).backward()
         
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
         
        if step % effective_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            predictor.model.zero_grad()
 
        #scheduler.step() """Moving it inside the if condition to avoid Pytorch warning"""
         
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
         
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step}: Current LR = {current_lr:.6f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")
    return mean_iou

def dice_bce_loss(pred, target, smooth=1.0):
    """
    Compute the Dice + BCE loss for the SAM2 model.
    
    Args:
        predictor: The SAM2ImagePredictor instance.
        target: The ground truth mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Computed loss value.
    """
    # Binary cross entropy loss
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    
    # Apply sigmoid to prediction for Dice calculation
    pred = torch.sigmoid(pred)
    
    # Flatten prediction and target tensors
    pred_flat = pred.view(-1)
    targets_flat = target.view(-1)
    
    # Calculate Dice coefficient
    intersection = (pred_flat * targets_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + targets_flat.sum() + smooth)
    dice_loss = 1 - dice_score
    
    # Combine losses (you can adjust the weighting)
    combined_loss = 0.5 * bce + 0.5 * dice_loss
    
    return combined_loss
def augment_image_and_mask(image, mask):
    # Random horizontal flip
    if random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Random vertical flip
    if random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # Random rotation (0, 90, 180, 270 degrees)
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    
    # Random brightness and contrast adjustment
    if random.random() > 0.5:
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Adjust brightness (V channel)
        hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.8, 1.2)
        # Clip values to valid range
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        # Convert back to RGB
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return image, mask

def validate(predictor, test_data, step, mean_iou):
    predictor.model.eval()
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            image, mask, input_point, num_masks = read_batch(test_data, visualize_data=False)
             
            if image is None or mask is None or num_masks == 0:
                print(f"Step {step}: Validating - Early return: image={image is not None}, mask={mask is not None}, num_masks={num_masks}")
                return mean_iou
     
            input_label = np.ones((num_masks, 1))
             
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                print(f"Step {step}: Validating - Early return: input_point type={type(input_point)}, input_label type={type(input_label)}")
                return mean_iou

            if input_point.size == 0 or input_label.size == 0:
                print(f"Step {step}: Validating - Early return: input_point size={input_point.size}, input_label size={input_label.size}")
                return mean_iou

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
             
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            
                print(f"Step {step}: Validating - Early return: unnorm_coords={unnorm_coords is not None}, labels={labels is not None}, unnorm_coords shape={unnorm_coords.shape if unnorm_coords is not None else 'None'}, labels shape={labels.shape if labels is not None else 'None'}")
                return mean_iou

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
 
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
 
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
 
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
 
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6)
                        - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()
 
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
 
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.1  # Match training
            loss = loss / accumulation_steps
 
            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)
             
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
 
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Current LR = {current_lr:.6f}, Valid_IoU = {mean_iou:.6f}, Valid_Seg Loss = {seg_loss:.6f}")
    return mean_iou

train_mean_iou = 0
valid_mean_iou = 0

for step in range(1, NO_OF_STEPS + 1):
    train_mean_iou = train(predictor, train_data, step, train_mean_iou)
    valid_mean_iou = validate(predictor, test_data, step, valid_mean_iou)


print("Training completed!")
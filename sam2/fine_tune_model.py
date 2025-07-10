from data_preparation import *  

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



# Specify the path to the SAM2 model checkpoint and configuration file
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"

# By initializing build_sam2 with these paths the core SAM2 model on the GPU is instantiated
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda:0")

# Class SAM2ImagePredictor is used to handle prompts and predictions
predictor = SAM2ImagePredictor(sam2_model)

# Training mode ensures that relevant layers can be fine-tuned when optimization routine starts
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)


scaler = GradScaler("cuda")  # For mixed precision training 
NO_OF_STEPS = 8000  # Number of training steps

# Creates de directory to save the fine-tuned model
def create_fine_tuned_model_dir():
    if not os.path.exists("../sam2/fine_tuned_models"):
        os.makedirs("../sam2/fine_tuned_models")
    print("Directory for fine-tuned models created or already exists.")

# Create the directory for fine-tuned models
create_fine_tuned_model_dir()
    
FINE_TUNED_MODEL_NAME = "cork_analizer_sam2"

#TODO: modify the learning rate (up)
optimizer = torch.optim.AdamW(params=predictor.model.parameters(),
                              lr=0.00005,  # Learning rate
                              weight_decay=1e-4)

# There are different learning rate schedulers available, here I'm using StepLR
#TODO: modify the step size and gamma (min seps 2000, gamma 0.6)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=900,
                                             gamma=0.6)  # Learning rate schedule
accumulation_steps = 8  # Gradient accumulation steps

# Training function

def train (predictor, train_data, step, mean_iou):
    with autocast("cuda"):
        image, mask, input_point, num_masks = read_batch(train_data, 
                                                         visualize_data=False)
        
        if image is None or mask is None or num_masks==0:
            return
        
        input_label = np.ones((num_masks,1))
        
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            return

        if input_point.size == 0 or input_label.size == 0:
            return

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point,
            input_label,
            box=None,
            mask_logits=None,
            normalize_coords=True
        )

        if unnorm_coords is None or labels is None or unnorm_box is None:
            return
        
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
        loss = seg_loss + score_loss * 0.05
         
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
         
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
         
        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()
 
        scheduler.step()
         
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
         
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step}: Current LR = {current_lr:.6f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")
    return mean_iou

def validate(predictor, test_data, step, mean_iou):
    predictor.model.eval()
    with autocast("cuda"):
        with torch.no_grad():
            image, mask, input_point, num_masks = read_batch(test_data, visualize_data=False)
             
            if image is None or mask is None or num_masks == 0:
                return
     
            input_label = np.ones((num_masks, 1))
             
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                return
     
            if input_point.size == 0 or input_label.size == 0:
                return
     
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
             
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                return
     
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
            loss = seg_loss + score_loss * 0.05
            loss = loss / accumulation_steps
 
            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)
             
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
 
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Current LR = {current_lr:.6f}, Valid_IoU = {mean_iou:.6f}, Valid_Seg Loss = {seg_loss:.6f}")
    return mean_iou

train_mean_iou = 0.0
valid_mean_iou = 0.0

for step in range(1, NO_OF_STEPS + 1):
    train_mean_iou = train(predictor, train_data, step, train_mean_iou)
    valid_mean_iou = validate(predictor, test_data, step, valid_mean_iou)

# Save the final model
final_model_path = f"../fine_tuned_models/{FINE_TUNED_MODEL_NAME}_final.pt"
torch.save(predictor.model.state_dict(), final_model_path)
print(f"Final model saved at: {final_model_path}")
print("Training completed!")
import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def convert_labelstudio_coco(json_file, output_dir):
    """
    Convert Label Studio export in COCO format to binary masks
    """
    print(f"Converting Label Studio COCO format from {json_file}...")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Load JSON file
    with open(json_file, "r") as f:
        coco_data = json.load(f)
    
    # Process images and annotations
    image_id_to_filename = {}
    for image in coco_data["images"]:
        image_id_to_filename[image["id"]] = image["file_name"]
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"Found {len(image_id_to_filename)} images and {len(image_annotations)} images with annotations")
    
    # Process each image
    successful_conversions = 0
    for image_id, filename in tqdm(image_id_to_filename.items(), desc="Processing images"):
        if image_id not in image_annotations:
            print(f"Warning: No annotations for image {filename}")
            continue
        
        # Find the corresponding image in the coco_data
        image_info = next((img for img in coco_data["images"] if img["id"] == image_id), None)
        if not image_info:
            continue
            
        width = image_info["width"]
        height = image_info["height"]
        
        # Create a blank mask
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw all segments for this image
        for ann in image_annotations[image_id]:
            if "segmentation" in ann:
                # COCO segmentation format can be RLE or polygon
                if isinstance(ann["segmentation"], dict):  # RLE format
                    # This would require pycocotools to decode
                    print(f"Warning: RLE format not supported for {filename}")
                else:  # Polygon format
                    for segment in ann["segmentation"]:
                        # Convert flat array to points
                        points = np.array(segment).reshape(-1, 2).astype(np.int32)
                        # Draw the polygon - OpenCV requires color as a tuple
                        cv2.fillPoly(mask, [points], (255, 0, 0))
        
        # Save the mask
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        mask_filename = os.path.join(masks_dir, f"{base_filename}_mask.png")
        Image.fromarray(mask).save(mask_filename)
        
        # Copy the image if available
        # For Label Studio's "COCO with images" export, images are usually in an 'images' folder
        json_dir = os.path.dirname(json_file)
        possible_image_paths = [
            os.path.join(json_dir, "images", filename),  # Label Studio's typical "COCO with images" structure
            os.path.join(json_dir, filename),            # Alternative location
            filename                                     # Absolute path
        ]
        
        image_found = False
        for img_path in possible_image_paths:
            if os.path.exists(img_path):
                # Determine output path - convert BMP to PNG
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                if filename.lower().endswith('.bmp'):
                    output_image_path = os.path.join(images_dir, f"{base_filename}.png")
                else:
                    output_image_path = os.path.join(images_dir, os.path.basename(filename))
                
                # Copy using PIL to handle different formats
                try:
                    img = Image.open(img_path)
                    # Save as PNG if original was BMP, otherwise keep original format
                    if filename.lower().endswith('.bmp'):
                        img.save(output_image_path, "PNG")
                        print(f"Converted {filename} from BMP to PNG")
                    else:
                        img.save(output_image_path)
                    image_found = True
                    successful_conversions += 1
                    break
                except Exception as e:
                    print(f"Error copying image {img_path}: {e}")
        
        if not image_found:
            print(f"Warning: Could not find image file {filename}. Tried paths:")
            for path in possible_image_paths:
                print(f"  - {path}")
    
    print(f"Conversion complete. Images and masks saved to {output_dir}")

def convert_voc(voc_dir, output_dir):
    """
    Convert PASCAL VOC format export from Label Studio to our required format
    """
    print(f"Converting PASCAL VOC format from {voc_dir}...")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Check for expected VOC directories
    segmentation_dir = os.path.join(voc_dir, "SegmentationClass")
    if not os.path.exists(segmentation_dir):
        segmentation_dir = os.path.join(voc_dir, "Segmentations")
    if not os.path.exists(segmentation_dir):
        segmentation_dir = os.path.join(voc_dir, "masks")
    
    if not os.path.exists(segmentation_dir):
        print(f"Error: Could not find segmentation directory in {voc_dir}")
        return
    
    # Find JPEGImages directory
    images_src_dir = os.path.join(voc_dir, "JPEGImages")
    if not os.path.exists(images_src_dir):
        images_src_dir = os.path.join(voc_dir, "images")
    
    if not os.path.exists(images_src_dir):
        print(f"Warning: Could not find images directory in {voc_dir}")
    
    # Process all mask files
    mask_files = list(Path(segmentation_dir).glob("*.png"))
    
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        # Read mask
        mask = np.array(Image.open(mask_file))
        
        # Convert to binary (any non-zero value becomes 255)
        binary_mask = np.zeros_like(mask)
        binary_mask[mask > 0] = 255, 3, 3
        
        # Save binary mask
        output_mask_path = os.path.join(masks_dir, mask_file.name)
        Image.fromarray(binary_mask.astype(np.uint8)).save(output_mask_path)
        
        # Copy corresponding image if available
        base_name = mask_file.stem
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(images_src_dir, f"{base_name}{ext}")
            if os.path.exists(image_path):
                output_image_path = os.path.join(images_dir, f"{base_name}{ext}")
                try:
                    img = Image.open(image_path)
                    img.save(output_image_path)
                    break
                except Exception as e:
                    print(f"Error copying image {image_path}: {e}")
    
    print(f"Conversion complete. Images and masks saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio exports to SAM training format")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to Label Studio export .json file (COCO JSON) or directory (VOC)")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Directory to save converted dataset")
    parser.add_argument("--format", type=str, choices=["auto", "coco", "voc"], default="auto", 
                       help="Format of the input: 'coco' for COCO JSON, 'voc' for PASCAL VOC directory")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output_dir
    
    # Auto-detect format if not specified
    if args.format == "auto":
        if os.path.isdir(input_path):
            args.format = "voc"
        elif input_path.endswith(".json"):
            args.format = "coco"
        else:
            print("Error: Could not auto-detect format. Please specify --format")
            return
    
    print("\nLabel Studio Export Conversion Tool for SAM Training")
    print("=====================================================")
    
    if args.format == "coco":
        print("Label Studio COCO Export Tips:")
        print("1. Make sure you chose 'COCO with images' when exporting")
        print("2. The input should be the path to the results.json file")
        print("3. Images should be in an 'images' folder alongside the JSON file")
        print("\nStarting conversion...\n")
        convert_labelstudio_coco(input_path, output_dir)
    elif args.format == "voc":
        print("Starting VOC format conversion...\n")
        convert_voc(input_path, output_dir)
        
print("\nAfter conversion, use prepare_csv_dataset.py to make the CVS file")

if __name__ == "__main__":
    main()
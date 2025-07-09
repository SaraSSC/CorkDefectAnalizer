import os
import pandas as pd
from pathlib import Path

def create_train_csv():
    """
    Creates a train.csv file mapping images to their corresponding masks.
    
    The CSV will contain two columns:
    - image_path: relative path to the image file
    - mask_path: relative path to the corresponding mask file
    """
    
    # Define paths
    images_dir = Path("./dataset/images")
    masks_dir = Path("./dataset/masks")
    output_csv = Path("./dataset/train.csv")
    
    # Check if directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Get all image files (both PNG and JPG)
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    if not image_files:
        raise ValueError("No .png nor .jpg images files found in the images directory")
    
    # Create lists to store the data
    image_id = []
    mask_id = []
    missing_masks = []
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image file
    for image_file in image_files:
        # Extract the base name without extension
        image_name = image_file.stem  # e.g., "4e6a7f61-rolha000017"
        
        # Create the corresponding mask filename
        mask_filename = f"{image_name}_mask.png"
        mask_file = masks_dir / mask_filename
        
        # Check if the corresponding mask exists
        if mask_file.exists():
            # Store relative paths
            image_id.append(f"{image_file.name}")
            mask_id.append(f"{mask_filename}")
            print(f"✓ Matched: {image_file.name} -> {mask_filename}")
        else:
            missing_masks.append(image_file.name)
            print(f"✗ Missing mask for: {image_file.name}")
    
    # Report missing masks
    if missing_masks:
        print(f"\nWarning: {len(missing_masks)} images don't have corresponding masks:")
        for missing in missing_masks:
            print(f"  - {missing}")
    
    # Create DataFrame
    if image_id:
        df = pd.DataFrame({
            'image_id': image_id,
            'mask_id': mask_id
        })
        
        # Create the output directory if it doesn't exist
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\nSuccess! Created {output_csv} with {len(df)} image-mask pairs")
        print(f"CSV file saved with columns: {list(df.columns)}")
        print(f"Sample entries:")
        print(df.head())
        
        return df
    else:
        print("No valid image-mask pairs found!")
        return None

def main():
    
    try:
       
        
        df = create_train_csv()
        
        if df is not None:
            print("\n" + "=" * 50)
            print("Dataset preparation completed successfully!")
        else:
            print("\n" + "=" * 50)
            print("Dataset preparation failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
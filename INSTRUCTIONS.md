# Training SAM on Custom Data

This project provides tools to fine-tune Meta's Segment Anything Model (SAM2.1) on custom segmentation datasets. 
The implementation includes dataset preparation, model training, and inference scripts.

# First step

## Only ignore this step if it's already installed in your system :

1.  Download the Microsoft Visual Studios Installer from from [Microsoft](https://visualstudio.microsoft.com/)
	1.  Open and run the .exe file for installation of Microsoft Visual Studios 
	2.  After installed, modify the installation and include " C++  for desktop development"
2.  Download and install BuildTools for Visual Studios from [Microsoft](https://visualstudio.microsoft.com/downloads/)
	1. After added to the Visual Studios Installer modify the installation and include "C++ for desktop development"
3.  Download and install Anaconda from [Anaconda](https://anaconda.org), don't forget to click on the checkbox to add the path to Windows Environments
4.  Download wget.exe from [eternallybored](https://eternallybored.org/misc/wget/ )
	1.  After the download, copy the file and past it in Windows/Sys32 folder in your drive
5. Download and install cuda-toolkit from [Nvidia](https://developer.nvidia.com/cuda-11-8-0-download-archive)

# Second step

## Environment Setup
(Note that you cannot ignore the [[#First step]] since SAM2.1 will not run without those programs )

1. Create and activate a conda environment:

```bash
#-n {name_you_want_for_your_env} //you can name it whatever you want
#-y it's for accepting automatically the prompt
conda create -n sam2_env python=3.10 -y

conda activate sam2_env

```

2. Install required packages:

  
```bash
#this will install the latest release version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python matplotlib pandas scipy pillow tqdm transformers accelerate pycocotools 

pip install label-studio

pip install gradio  # For web interface

```
 

Alternatively, you can install all dependencies from the requirements.txt file:

  
```bash

pip install -r requirements.txt

```


## Clone the Github repository to and folder/local of choice or download via their provided GUI interface

- if by  GUI download, just unzip the folder to the location you want
- if by using git commands 

```bash 

git clone https://github.com/SaraSSC/CorkDefectAnalizer.git

```
## Or start from scratch

Install sam2 

```bash

git clone https://github.com/facebookresearch/sam2.git

```
Move inside the `sam2` folder:

```bash

cd sam2
pip install -e . #or pip install -e ".[dev]"

```
Move inside checkpoints and install the checkpoints 

```bash

cd checkpoints
download_ckpts.sh #or download_ckpts.bat

```


## Dataset Preparation

### Option A: Using Raw Images and Masks

If you already have image, masks and train.csv files just place then in the `./dataset` in their respective folders
- `./dataset/images` for the images 
- `./dataset/masks` for their masks
- `./dataset` for the `train.csv` file

### Option B: Converting from Label Studio


If you're using Label Studio for annotations:

  
1. Export your project in **"COCO with images"** format (which includes both annotations and images[in  .bmp format])

2. Extract the ZIP file from Label Studio to  `./sam2/label_studio_exports/`

   - This will create a JSON file (typically `result.json`) and an `images` folder

   - **Important**: Keep the original folder structure - the JSON file and `images` folder must be in the same directory

1. Use the `prepare_dataset.py` script to convert the exports to our format:

   - This python file is made to handle coco and pascal voc formats

```bash

--format {option: coco or voc}

```


```bash
# For Label Studio's COCO export format

# Point to the JSON file; the script will automatically find the images folder next to it

python create_dataset.py --input /path/to/label_studio_export/result.json --output_dir ./dataset --format coco

```


After running the conversion script, you'll have:


```

dataset/

├── images/              # Images copied from the Label Studio export and                                     converted to .png extension

│   ├── image1.png

│   ├── image2.png

│   └── ...

└── masks/               # Generated binary masks from the polygon annotations

    ├── image1.png

    ├── image2.png

    └── ...

```

  2. Use the prepare_csv_dataset.py to create the .csv file that contains the mapping 
	  A CSV file will map each image to its corresponding segmentation mask, ensuring proper indexing for SAM2 training.
	  
```bash

python create_train_csv_dataset.py

```

Final structure after runNing:

```
dataset/

├── images/             

│   ├── image1.png

│   ├── image2.png

│   └── ...

├── masks/              

|   ├── image1.png

|   ├── image2.png

|   └── ...

└── train.csv


```

Then run the `data_preparation.py` script to prepare the dataset for training:

```bash

python data_preparation.py

```
This script will read the images and masks, resize them to 1024x1024, and generate random points on the regions of interest (ROIs) in the masks. The output will be a batch of images, binary masks, and points ready for training.

## Fine-tuning SAM2.1

After preparing the dataset, you can fine-tune SAM2.1 using the `fine_tune_model.py` script:

```bash

python fine_tune_model.py

```
This script will load the SAM2.1 model, prepare the dataset, and start training. It will save checkpoints and log training progress.
It will also open a image visualization window to a sample image from the dataset, just to check if the data is being loaded correctly. You need to close it so the model can start the training. You can close it by pressing `q` or `esc` or by clicking in the X button.

## Inference
To run inference on new images using the fine-tuned model, use the `inference_fine_tuned.py` script:

```bash

python inference_fine_tuned.py

```
If you are using a low memory GPU, run instead:

```bash

python inference_low_gpu.py

```

## Tips for Training SAM

  
1. **Data Preparation**:

   - Ensure your masks are binary (foreground=1, background=0)

   - Make sure image and mask sizes match

   - Provide diverse examples for better generalization


1. **Training Parameters**:

   - Start with a small learning rate (1e-5 to 1e-6)

   - Use a relatively small batch size due to model size

   - Train for at least 10 epochs to see meaningful improvement
   

3. **Hardware Requirements**:

   - SAM2.1-huge requires at least 24GB of GPU memory

   - For systems with less memory, consider using a smaller variant
   

4. **Performance Optimization**:

   - If training is slow, consider resizing your images to a consistent resolution

   - Use mixed precision training by enabling `torch.cuda.amp`

## Annotation Guidelines

### Required Annotation Format

SAM training requires binary segmentation masks with these specifications:


1. **File Format**:

   - PNG files (recommended for lossless compression)

   - Each mask must have the same filename as its corresponding image (with .png extension)

   - Example: `image1.jpg` → `image1_mask.png`
			  `image1.png` → `image1_mask.png`

  

2. **Mask Properties**:

   - Single-channel (grayscale) images

   - Binary values: 0 for background, 255 (or any non-zero value) for foreground

   - Same dimensions as the input images

  

3. **Multiple Objects**:

   - For multiple objects in one image, you can use separate mask files for each object

   - Alternatively, use instance segmentation with different pixel values for each object



### Annotation Tools


Here are some recommended tools for creating segmentation masks:

  
1. **[CVAT](https://cvat.org/)**:

   - Free, open-source web-based annotation tool

   - Supports polygon, brush, and semi-automatic segmentation

   - Can export directly as binary masks

  

2. **[LabelMe](https://github.com/wkentaro/labelme)**:

   - Simple Python tool for polygon annotations

   - Lightweight and easy to use locally

   - Exports as JSON that can be converted to masks

  

3. **[Supervisely](https://supervise.ly/)**:

   - Comprehensive platform with free tier

   - Advanced annotation features including AI assistance

   - Supports various export formats including masks

  

4. **[Roboflow](https://roboflow.com/)**:

   - Good for managing datasets with free tier

   - Pre-processing and augmentation tools

   - Can export in various formats

  

5. **[Label Studio](https://labelstud.io/)**:

   - Open-source data labeling tool with both cloud and self-hosted options

   - **Project Setup for SAM Training**:

     1. When creating a project, choose **"Image Segmentation with Polygons"** (preferred over brush-based segmentation)

     2. Configure your labels for the objects you want to segment

     3. Use the polygon tool to create precise boundaries around your objects

  

   - **Export Instructions for SAM Training**:

     1. Go to your project and click "Export"

     2. Select **"COCO with images"** format (recommended) which includes both annotations and image files

     3. After export, extract the ZIP file which will contain a JSON file and an 'images' folder

     4. Convert the export to our required format using the converter script:


# RenSeg: Kidney CT Segmentation and Classification

This repository contains the code accompanying the paper:

**RenSeg: Leveraging Unsupervised Segmentation using Localization and Contour-Guided Quickshift for Renal Calculi and Carcinoma Segmentation and Classification**

---

## Features

- YOLOv8 for kidney and aorta localization  
- Contour-guided Quickshift segmentation  
- Cropping and resizing to 224×224  
- Preparing data for classification models (MobileNetV2, VGG19, ResNet50)

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sarweralam867/RenSeg.git
cd RenSeg
pip install -r requirements.txt
````

For CPU-only environments:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Datasets

Two datasets are required:

1. **Detection Dataset (for YOLOv8 training)**

   * Prepare using [Roboflow](https://roboflow.com/)

   * Annotate kidneys and aorta

   * Export in YOLOv8 format and place `data.yaml` inside the `data/` directory

   * Directory structure:

     ```
     data/
     ├── data.yaml
     ├── train/
     │   ├── images/
     │   └── labels/
     └── val/
         ├── images/
         └── labels/
     ```

   * Used when running `train_main.py`

2. **Main CT Dataset (for segmentation/inference)**

   * Original CT scans (unannotated)

   * Place CT test images inside `data/test/` before running inference:

     ```
     data/
     └── test/
         ├── img1.png
         ├── img2.png
         └── ...
     ```

   * Processed outputs will be saved in the `outputs/` folder

---

## Usage

### 1. Training YOLOv8

```bash
python train_main.py
```

### 2. Running Segmentation / Inference

```bash
python segment_main.py
```

Segmented and resized images will be saved in the `outputs/` folder.

---

## Project Structure

```
RENSEG/
├── __init__.py
├── .gitignore
├── config.py
├── inference.py
├── LICENSE
├── postprocessing.py
├── preprocessing.py
├── requirements.txt
├── segment_main.py
├── train_main.py
├── training.py
├── utils.py
├── data/                # Datasets go here
│   ├── data.yaml        # YOLOv8 dataset config
│   ├── train/           # Training images + labels
│   ├── val/             # Validation images + labels
│   └── test/            # CT scans for segmentation/inference
├── checkpoints/         # Model weights after training
└── outputs/             # Segmented images
```

---

## Reference

If this code is used, please cite:

**RenSeg: Leveraging Unsupervised Segmentation using Localization and Contour-Guided Quickshift for Renal Calculi and Carcinoma Segmentation and Classification**
Authors: Farhan Faruk, H. M. Sarwer Alam, Rafeed Rahman, Md. Golam Rabiul Alam, Junho Jeong, Md. Kabir Hossain
Journal: IEEE Journal of Biomedical and Health Informatics
DOI: 10.1109/JBHI.2025.3629580

Plain text:
F. Faruk et al., "RenSeg: Leveraging Unsupervised Segmentation Using Localization and Contour-Guided Quickshift for Renal Calculi and Carcinoma Segmentation and Classification," in IEEE Journal of Biomedical and Health Informatics, doi: 10.1109/JBHI.2025.3629580.
keywords: {Kidney;Image segmentation;Accuracy;Kidney stones;Diseases;Tumors;Computed tomography;Transformers;Bioinformatics;Annotations;Calculi;carcinoma;classification;IoU;renal;segmentation},


BibTeX:
@ARTICLE{11286221,
  author={Faruk, Farhan and Alam, H. M. Sarwer and Rahman, Rafeed and Alam, Md. Golam Rabiul and Jeong, Junho and Hossain, Md. Kabir and Uddin, Jia},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={RenSeg: Leveraging Unsupervised Segmentation Using Localization and Contour-Guided Quickshift for Renal Calculi and Carcinoma Segmentation and Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Kidney;Image segmentation;Accuracy;Kidney stones;Diseases;Tumors;Computed tomography;Transformers;Bioinformatics;Annotations;Calculi;carcinoma;classification;IoU;renal;segmentation},
  doi={10.1109/JBHI.2025.3629580}}




**Welcome to Week 3**

***Problem Statement***

Students are required to develop a deep learning–based system that can automatically segment brain tumors from MRI scans. Using the BraTS dataset, which provides multi-modal MRI volumes and corresponding ground-truth segmentation masks, students must preprocess the data by loading MRI files, converting 3D volumes into 2D slices, normalizing and resizing the images, and preparing paired input–mask datasets.

Students must then implement a U-Net segmentation model using either TensorFlow/Keras or Pytorch, train the model on the prepared data, and evaluate its performance using metrics such as Dice Score and Intersection over Union (IoU). The final output should include the trained model, quantitative evaluation results, and visualizations comparing predicted segmentation masks with the ground-truth masks.

Link to the dataset: [BRaTS 2021 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

The theory guide for the project can be found here: [Theory Guide](https://docs.google.com/document/d/1_FPFWqnD06kb9QH93hEePEFe_FTZRMp1MgsGIDm8nDc/edit?tab=t.0#heading=h.kxvpgxpt58uh)


# My description:


# Brain Tumor Segmentation using U-Net (BraTS 2021)

---

## Project Overview

This project implements a deep learning–based system for automatic brain tumor segmentation from MRI scans.  
A U-Net architecture is trained using the BraTS 2021 dataset to predict tumor regions from MRI slices.

The work is done as part of **Week 3** of the Winter in Data Science program and builds upon the fundamentals covered in earlier weeks such as NumPy, image handling, and PyTorch basics.

---

## Problem Statement

The objective of this project is to design and train a neural network that can accurately segment brain tumors from MRI images.  
The model takes MRI slices as input and generates corresponding segmentation masks that highlight tumor regions.

---

## Dataset

- **Dataset Name:** BraTS 2021 (Task 1)
- **Type:** Multi-modal MRI scans with ground truth segmentation masks
- **Source:** Kaggle

Dataset Link:  
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

---

## Tools and Libraries Used

- Python
- NumPy
- Matplotlib
- NiBabel
- PyTorch
- Google Colab

---

## Approach

The overall workflow of the project is as follows:

1. Load MRI volumes and segmentation masks using NiBabel  
2. Convert 3D MRI volumes into 2D slices  
3. Normalize and resize the MRI slices  
4. Prepare paired image-mask datasets  
5. Implement a U-Net based segmentation model  
6. Train the model using Binary Cross Entropy loss  
7. Evaluate performance using Dice Score  
8. Visualize predicted masks against ground truth  
9. Save the trained model for future use  

---

## Model Architecture

- The model is based on a **U-Net architecture**
- Consists of:
  - Encoder blocks for feature extraction
  - Bottleneck layer
  - Decoder blocks with skip connections
- Final output layer uses a sigmoid activation for binary segmentation

---

## Evaluation Metric

- **Dice Score** is used to evaluate segmentation performance
- Dice Score measures overlap between predicted and ground truth masks

---

## Training Details

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: Binary Cross Entropy Loss
- Input Size: 128 × 128
- Training performed on Google Colab (GPU if available)

---

## Code Implementation

The complete implementation including:
- Data loading
- Preprocessing
- Model definition
- Training loop
- Evaluation
- Visualization
- Model saving

is available in the Google Colab notebook below.

Code Link:  
https://colab.research.google.com/drive/1I-oKdAMHRKto5pjCfpun0Kw68ePQRKSk?usp=sharing

---

## Output

- Trained U-Net model
- Dice score evaluation
- Visual comparison of:
  - MRI input
  - Ground truth mask
  - Predicted mask
- Saved model file (`.pth` format)

---

## Learning Outcome

Through this project, the following concepts were learned and applied:

- Medical image handling using NiBabel
- MRI data preprocessing
- U-Net architecture for segmentation
- PyTorch dataset and dataloader usage
- Training and evaluating segmentation models
- Understanding Dice Score metric

---

## Conclusion

This project demonstrates how deep learning can be effectively used for medical image segmentation tasks.  
The U-Net model provides a reliable baseline for brain tumor segmentation and can be further improved with more training, data augmentation, and multi-modal MRI inputs.

---


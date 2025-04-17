# Thai Dishes Image Classification using EfficientNet-B0

This repository contains code for building an image classification model to identify Thai dishes from images using PyTorch and a pre-trained EfficientNet-B0 model.

## 🥘 Project Overview

This project includes:
- Image preprocessing and resizing
- Train-test data splitting
- Data augmentation for robustness
- Training a deep learning model (EfficientNet-B0)
- Evaluation using multiple metrics
- Visualization of predictions
- Saving model and plots

## 📁 Dataset

The dataset should be structured in a directory where each subfolder represents a class (dish). Example structure:

thai_dishes_images/ ├── Pad_Thai/ │ ├── img1.jpg │ └── ... ├── Tom_Yum/ │ ├── img1.jpg │ └── ...


Paths used:
- Input images: `/content/drive/MyDrive/thai_dishes_images`
- Processed & split data: `/content/drive/MyDrive/thai_dishes_split`
- Trained model: `/content/drive/MyDrive/thai_dishes_model_efficientnet.pth`
- Plot output: `/content/drive/MyDrive/training_validation_plots.png`

## 🚀 Pipeline

### 1. Image Preprocessing
- Resize images to 224x224
- Convert all images to RGB
- Save processed images in a new directory

### 2. Train-Test Split
- Randomly select 2 images from each class for testing
- The rest are used for training

### 3. Data Augmentation
- Applied only to training set:
  - Random crop, flips, rotations, color jitter, grayscale, etc.
- Validation set is center-cropped and resized

### 4. Model Training
- Pre-trained EfficientNet-B0 is used
- Final classifier modified to match number of classes
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- 50 Epochs

### 5. Evaluation Metrics
- Accuracy
- F1 Score (Weighted)
- Confusion Matrix
- Mean Absolute Error (MAE)

### 6. Visualization
- Visual inspection of predictions
- Training and validation loss and accuracy plots

## 🧠 Dependencies

Install required libraries:
```bash
pip install torch torchvision scikit-learn matplotlib pillow
📊 Results
After training, the model prints and visualizes:

Accuracy and F1 Score

Confusion Matrix

Sample predictions vs. ground truth

Plots of training/validation loss and accuracy

💾 Model Saving
The trained model is saved as:
/content/drive/MyDrive/thai_dishes_model_efficientnet.pth

🖼️ Example Output
Model predictions on random test images

Loss and Accuracy plots saved to:

/content/drive/MyDrive/training_validation_plots.png
✍️ Author
Nimra – AI Researcher & Deep Learning Enthusiast

📜 License
This project is open-source and available for research and educational purposes.

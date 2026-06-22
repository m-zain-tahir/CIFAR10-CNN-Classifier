# CIFAR-10 Image Classifier — CNN with Data Augmentation

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> A deep learning image classification project that trains and compares 
> two CNN models on the CIFAR-10 dataset — one without augmentation and 
> one with custom data augmentation — complete with a GUI predictor app.

---

## Project Overview

This project was developed as part of the Artificial Intelligence course  
at the **Department of Software Engineering, University of Lahore**.

We built, trained, and evaluated Convolutional Neural Networks (CNNs) on 
the CIFAR-10 dataset — 60,000 images across 10 classes. The core goal was 
to compare model performance with and without data augmentation, and measure 
the real impact augmentation has on accuracy.

**Authors:**
- Muhammad Zain Tahir
- Huzaifa Tanveer

---

## Dataset — CIFAR-10

| Property | Details |
|---|---|
| Total Images | 60,000 (32×32 RGB) |
| Training Set | 50,000 images |
| Test Set | 10,000 images |
| Classes | 10 |

**10 Classes:**
`Airplane` `Automobile` `Bird` `Cat` `Deer` `Dog` `Frog` `Horse` `Ship` `Truck`

---

## Features

### Two CNN Models Trained & Compared
- **CNN Plain** — trained on original 50,000 images
- **CNN Augmented** — trained on 150,000 images (3× via custom augmentation)

### Custom Data Augmentation (NumPy-based)
Instead of using ImageDataGenerator, we implemented manual augmentation 
for better Windows compatibility:
- Horizontal flip of each image
- Row-shift simulation (rotation effect)
- Result: 50,000 → 150,000 training samples

### CNN Architecture
Deep 6-layer convolutional network:
```
Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.30)
Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.40)
Flatten → Dense(256) → BN → Dropout(0.50)
Dense(128) → Dropout(0.30)
Dense(10, softmax)
```
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping + ReduceLROnPlateau

### GUI Predictor App
A desktop application built with Tkinter:
- Upload any image from your computer
- Get real-time prediction with confidence score
- Visual probability bar for all 10 classes
- Dark-themed professional UI

### Output Graphs Generated
| File | Description |
|---|---|
| `fig1_sample_images.png` | Sample images from each class |
| `fig2_class_distribution.png` | Class balance in dataset |
| `fig3_augmentation_comparison.png` | Original vs augmented images |
| `fig4_training_curves.png` | Accuracy & loss over epochs |
| `fig5_cm_plain.png` | Confusion matrix — Plain CNN |
| `fig6_cm_aug.png` | Confusion matrix — Augmented CNN |
| `fig7_model_comparison.png` | Side-by-side model comparison |

---

## Project Structure

```
CIFAR10-CNN-Classifier/
│
├── train_once.py            # Train both CNN models + save
├── results.py               # Evaluate models + generate all graphs
├── gui_predictor.py         # Tkinter GUI prediction app
├── main_project_OLD.py      # Earlier version (reference only)
│
└── outputs/
    ├── fig1_sample_images.png
    ├── fig2_class_distribution.png
    ├── fig3_augmentation_comparison.png
    ├── fig4_training_curves.png
    ├── fig5_cm_plain.png
    ├── fig6_cm_aug.png
    └── fig7_model_comparison.png
```

> Note: `.keras` model files are not included in this repo due to size 
> (each ~10MB). Run `train_once.py` to generate them locally.

---

## How to Run

### Step 1 — Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

### Step 2 — Train the Models
```bash
python train_once.py
```
This will:
- Download CIFAR-10 automatically
- Build augmented dataset (150,000 images)
- Train both CNN models (~30-40 minutes)
- Save models to `outputs/` folder

### Step 3 — Generate Results & Graphs
```bash
python results.py
```
This will:
- Load both trained models
- Evaluate accuracy, precision, recall, F1-score
- Generate all 7 output graphs

### Step 4 — Run GUI Predictor
```bash
python gui_predictor.py
```
- Upload any image
- Click Predict
- See results instantly

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Primary language |
| TensorFlow / Keras | CNN model building & training |
| NumPy | Data manipulation & augmentation |
| Matplotlib | Graph generation |
| Seaborn | Confusion matrix visualization |
| scikit-learn | Metrics (F1, Precision, Recall) |
| Tkinter | GUI application |
| Pillow | Image loading in GUI |

---

## Academic Context

**Course:** Artificial Intelligence  
**Department:** Software Engineering  
**University:** University of Lahore  
**Semester:** Spring 2026

---

## Authors

**Muhammad Zain Tahir**  
B.S. Software Engineering — University of Lahore  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/muhammad-zain-tahir07)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/m-zain-tahir)

**Huzaifa Tanveer**  
B.S. Software Engineering — University of Lahore

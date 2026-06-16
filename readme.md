# DermaDetect — Skin Disease Classification FYP

A deep learning Final Year Project for automated classification of **Tinea** (10 subtypes) and **Vitiligo** skin conditions using transfer learning on ImageNet-pretrained CNN backbones.

---

## Overview

DermaDetect trains and compares three state-of-the-art convolutional neural network architectures (MobileNetV2, VGG16, ResNet50) on a curated dermatological image dataset. The goal is to assist in early, non-invasive identification of common skin conditions, bridging the gap between clinical expertise and accessible diagnostics.

**Task:** 11-class multiclass image classification  
**Conditions:** Vitiligo + 10 Tinea subtypes  
**Best Model:** VGG16 — **91.76% test accuracy**

---

## Classes (11 total)

| Index | Class              |
|-------|--------------------|
| 0     | Vitiligo           |
| 1     | tinea-body         |
| 2     | tinea-face         |
| 3     | tinea-foot-dorsum  |
| 4     | tinea-foot-plantar |
| 5     | tinea-foot-webs    |
| 6     | tinea-groin        |
| 7     | tinea-hand-dorsum  |
| 8     | tinea-palm         |
| 9     | tinea-scalp        |
| 10    | tinea-versicolor   |

---

## Results

All three models were fine-tuned in two phases (head training + backbone fine-tuning, up to 40 epochs total). Final metrics on the held-out **test set**:

| Metric            | MobileNetV2 | VGG16      | ResNet50  |
|-------------------|-------------|------------|-----------|
| **Accuracy**      | 89.93%      | **91.76%** | 73.00%    |
| Precision (W)     | 91.20%      | **92.04%** | 75.05%    |
| Precision (M)     | 75.12%      | **78.04%** | 42.27%    |
| Recall (W)        | 89.93%      | **91.76%** | 73.00%    |
| Recall (M)        | 77.90%      | **81.33%** | 51.89%    |
| F1-Score (W)      | 90.29%      | **91.68%** | 72.68%    |
| F1-Score (M)      | 75.94%      | **78.79%** | 43.85%    |

> **(W) = Weighted average, (M) = Macro average*

**VGG16** achieved the highest performance across all metrics and is the recommended model for deployment.

---

## Pipeline

### 1. Data Preparation
- Dataset split into `Train / Val / Test` directories (one sub-folder per class)
- Images resized to **224 × 224** pixels (standard for all three backbones)
- Batch size: **25**
- **Balanced class weights** computed via `sklearn.utils.class_weight.compute_class_weight` to handle class imbalance

### 2. Preprocessing
- Pixel values scaled to `[0, 1]` (`uint8 ÷ 255`)
- ImageNet normalization applied per channel:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std:  `[0.229, 0.224, 0.225]`

### 3. Model Architecture
All models follow an identical head design on top of the frozen ImageNet backbone:

```
Backbone (ImageNet pre-trained, frozen in Phase 1)
  └─ GlobalAveragePooling2D
      └─ Dense(256, relu) + Dropout
          └─ Dense(11, softmax)   ← one logit per class
```

- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Phase 1 LR:** `1e-4` (head only)
- **Phase 2 LR:** `1e-5` (last backbone layers + head)

### 4. Training Strategy (Two-Phase)

| Phase | Epochs (max) | Layers Trained     | Learning Rate |
|-------|--------------|--------------------|---------------|
| 1     | 25           | Classification head | `1e-4`        |
| 2     | 15           | Last backbone block + head | `1e-5` |

**Callbacks used:**
- `EarlyStopping` — monitors `val_loss`, patience = 5, restores best weights
- `ModelCheckpoint` — saves best checkpoint per phase
- `ReduceLROnPlateau` — reduces LR on plateau

### 5. Evaluation
- `classification_report` per class (precision, recall, F1)
- Weighted and macro-averaged aggregate metrics
- Confusion matrix visualization
- Training curves (accuracy & loss) across both phases

---

## Project Structure

```
DermaDetect-ANd-Advisor-FYP-/
│
├── main_FYP_classified_40.ipynb  # Full pipeline: all 3 models, Phase 1 + Phase 2 training
├── main_FYP_classified_25.ipynb  # Phase 1 only (25-epoch head training)
├── pyproject.toml                # Python project configuration (uv package manager)
├── uv.lock                       # Dependency lock file
└── readme.md                     # This file
```

> **Note:** The `Train/`, `Val/`, and `Test/` dataset directories are hosted on Google Drive and are not included in this repository.

---

## Setup & Requirements

This project was run on **Google Colab** with GPU acceleration and datasets stored on Google Drive.

**Core dependencies:**
- Python ≥ 3.13
- TensorFlow / Keras
- NumPy ≥ 2.4.4
- scikit-learn
- Matplotlib

**Install with uv:**
```bash
pip install uv
uv sync
```

**To run on Colab:**
1. Mount your Google Drive with the dataset at `MyDrive/Classified_dataset/`
2. Open `main_FYP_classified_40.ipynb` in Google Colab
3. Run all cells in order

---

## Key Hyperparameters

| Parameter         | Value    |
|-------------------|----------|
| Image size        | 224 × 224 |
| Batch size        | 25       |
| Phase 1 epochs    | 25 (max) |
| Phase 2 epochs    | 15 (max) |
| Phase 1 LR        | 1e-4     |
| Phase 2 LR        | 1e-5     |
| Early stopping patience | 5 epochs |
| Random seed       | 42       |
| Class weighting   | Balanced |

---


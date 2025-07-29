# CNN Image Classifier (DA6401 - Assignment 2)

This repository contains two parts:

- `Part A`: A CNN model built from scratch on Fashion MNIST.
- `Part B`: Transfer Learning using pre-trained **VGG16** on Fashion MNIST.

---

## 📁 Directory Structure

cnn-image-classifier/
│
├── README.md
│
├── partA/
│ ├── CNN_PartA.ipynb # Jupyter notebook for basic CNN from scratch
│ └── train.py # Command-line script for Part A
│
└── partB/
├── CNN_PartB.ipynb # Jupyter notebook using VGG16 transfer learning
└── train.py # Command-line script for Part B

yaml
Copy
Edit

---

## 🔧 Setup Instructions

1. Install dependencies (you can use conda or pip):
    ```bash
    pip install tensorflow wandb numpy matplotlib
    ```

2. Login to [Weights & Biases](https://wandb.ai/) for logging experiments:
    ```bash
    wandb login
    ```

---

## ▶️ Running Code

### Part A: Basic CNN

- **Train using script:**
    ```bash
    python partA/train.py --epochs 10 --batch_size 64 --learning_rate 0.001
    ```

- **Or open Jupyter Notebook:**
    ```bash
    jupyter notebook partA/CNN_PartA.ipynb
    ```

---

### Part B: Transfer Learning with VGG16

- **Train using script:**
    ```bash
    python partB/train.py --epochs 10 --batch_size 32 --learning_rate 0.0001
    ```

- **Or open Jupyter Notebook:**
    ```bash
    jupyter notebook partB/CNN_PartB.ipynb
    ```

---

## 🧪 Model Evaluation

Both scripts and notebooks print:

- Training & Validation Accuracy/Loss per epoch.
- Final Test Accuracy.
- Automatically log confusion matrix and metrics to Weights & Biases dashboard.

---

## 📊 Results Overview

| Model          | Accuracy (approx.) |
|----------------|--------------------|
| Basic CNN      | 88–90%             |
| VGG16 Transfer | 91–94%             |

---

## 📎 Notes

- Uses **Fashion MNIST** dataset (grayscale 28×28 images).
- Part B resizes input images to 224×224×3 as required by VGG16.
- WandB logging is optional but recommended for visualization.

---

## 🧑‍💻 Author

**Lokesh Talamala**  
DA6401 - Deep Learning Assignment 2  
Indian Institute of Technology Madras
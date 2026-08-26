# NovaMind — Neural Network from Scratch

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Built with NumPy](https://img.shields.io/badge/Built%20with-NumPy%20Only-4c1)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.91%25-4c1)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A multi-layer perceptron built entirely in **NumPy** — no PyTorch, no TensorFlow, no autograd. Every forward pass, backward pass, and weight update is written by hand.

Trained on MNIST (60k images). Achieves **97.91% test accuracy**.

---

## Results

| Metric | Value |
|---|---|
| Train Accuracy | 99.69% |
| Test Accuracy | 97.91% |
| Overfitting Gap | 1.78% |
| Epochs | 20 |
| Optimizer | Mini-batch SGD + L2 Regularization |

---

## Architecture

```
Input (784)
    ↓
Dense (784 → 128) + ReLU
    ↓
Dense (128 → 64) + ReLU
    ↓
Dense (64 → 10) + Softmax
    ↓
Output (10 classes)
```

Weights initialized with **Xavier initialization**: `W = randn * sqrt(1 / input_dim)`

---

## What's Built from Scratch

- **Forward pass** — matrix multiplication, bias addition, activation functions
- **Backpropagation** — full chain rule implementation through every layer
- **ReLU** — forward `max(0, z)` and backward `d_out * (input > 0)`
- **Softmax** — numerically stable implementation with combined Softmax + CrossEntropy gradient `(predictions - y_true) / batch_size`
- **CrossEntropy Loss** — `- mean(sum(y_true * log(clip(pred)), axis=1))`
- **Mini-batch SGD** — shuffles data every epoch, processes in batches of 64
- **L2 Regularization** — weight decay term `λ * W` added to gradient during update; biases not regularized
- **Xavier Initialization** — prevents vanishing/exploding gradients at initialization
- **MNIST data pipeline** — downloads, parses IDX binary format, normalizes, one-hot encodes

---

## Visualizations

### Training Curves
![Training Curves](plots/training_curves.png)

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### Prediction Grid
![Prediction Grid](plots/prediction_grid.png)
---

## Project Structure

```
novamind/
├── src/
│   ├── activations.py      # ReLU, Softmax
│   ├── layers.py           # Dense layer (forward + backward)
│   ├── losses.py           # CrossEntropyLoss
│   ├── network.py          # Network base class + MLP
│   └── utils.py            # MNIST downloader, parser, normalizer
├── config.py               # Hyperparameters
├── train.py                # Training loop
├── evaluate.py             # Test accuracy on saved weights
├── visualize.py            # Loss curves, confusion matrix, prediction grid
└── plots/                  # Generated visualizations
```

---

## How to Run

**Install dependencies:**
```bash
pip install numpy matplotlib
```

**Train:**
```bash
python train.py
```

**Evaluate:**
```bash
python evaluate.py
```

**Generate visualizations:**
```bash
python visualize.py
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | 0.1 |
| Batch Size | 64 |
| Epochs | 20 |
| L2 Lambda | 0.0001 |
| Hidden Layer 1 | 128 |
| Hidden Layer 2 | 64 |
| Random Seed | 42 |

---

## Key Design Decisions

**Softmax handles the backward pass for CrossEntropy.** The combined gradient of Softmax + CrossEntropy simplifies to `(predictions - y_true) / batch_size`, so `CrossEntropyLoss` has no `backward` method — Softmax absorbs it entirely.

**Biases are not L2 regularized.** Regularizing biases provides no benefit and can hurt convergence. Only weights are penalized.

**Weight saving is fully dynamic.** The training loop finds Dense layers using `hasattr(layer, 'W')` rather than hardcoding layer indices, so the architecture can be changed without breaking the save/load logic.
# NovaMind

A neural network built from scratch using only NumPy. No PyTorch, no TensorFlow, no autograd — every forward pass, backward pass, and weight update is written by hand.

Trained on MNIST. Gets 97.91% test accuracy.

---

## What I implemented

- Dense layers with forward and backward pass
- ReLU and Softmax activations
- Cross-entropy loss
- Mini-batch SGD with L2 regularization
- Xavier weight initialization
- MNIST data pipeline — downloads and parses the IDX binary format, normalizes, one-hot encodes

## Architecture

```
Input (784) → Dense → ReLU → Dense → ReLU → Dense → Softmax → Output (10)
Layer sizes:   784 → 128 → 64 → 10
```

## Results

| | |
|---|---|
| Train accuracy | 99.69% |
| Test accuracy | 97.91% |
| Epochs | 20 |
| Batch size | 64 |
| Learning rate | 0.1 |
| L2 lambda | 0.0001 |

## How to run

```bash
pip install numpy matplotlib
python train.py      # trains and saves weights
python evaluate.py   # prints test accuracy
python visualize.py  # generates plots
```

## One thing worth noting

The Softmax and CrossEntropy backward passes are combined. The gradient simplifies to `(predictions - y_true) / batch_size`, so CrossEntropyLoss has no backward method — Softmax handles it entirely. This only becomes obvious when you write the math yourself.

## Interactive Demo

The `docs/` folder has a browser demo (draw a digit, model predicts it) built for Hack Club Stardance. It was AI-generated and is not part of the core project. The trained weights are exported to JSON and the forward pass runs in JavaScript.
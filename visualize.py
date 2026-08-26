import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_mnist
from src.network import MLP
from config import INPUT_DIM, HIDDEN_1, HIDDEN_2, OUTPUT_DIM

def load_model():
    model = MLP(INPUT_DIM, HIDDEN_1, HIDDEN_2, OUTPUT_DIM)
    data = np.load('model_weights.npz')
    dense_list = [layer for layer in model.layers if hasattr(layer, 'W')]
    for idx, layer in enumerate(dense_list, start=1):
        layer.W = data[f'W{idx}']
        layer.b = data[f'b{idx}']
    return model

def plot_training_curves(history_file, final_test_acc):
    data = np.load(history_file)
    loss = data['loss']
    train_acc = data['train_acc']
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, loss, 'b-', label='Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_acc, 'g-', label='Training Accuracy')
    ax2.axhline(y=final_test_acc, color='r', linestyle='--', label=f'Final Test Acc: {final_test_acc:.2f}%')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Annotate cells with counts
    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i][j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i][j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()

def plot_prediction_grid(X_test, y_true, y_pred):
    wrong_idx = np.where(y_pred != y_true)[0]
    correct_idx = np.where(y_pred == y_true)[0]
    
    np.random.seed(42)
    selected_wrong = np.random.choice(wrong_idx, size=10, replace=False)
    selected_correct = np.random.choice(correct_idx, size=30, replace=False)
    
    # Combine wrong and correct indices
    selected_idx = np.concatenate([selected_wrong, selected_correct])
    
    fig, axes = plt.subplots(5, 8, figsize=(12, 8))
    fig.suptitle("Model Predictions: 10 Wrong (Red) | 30 Correct (Green)", fontsize=16, y=1.02)
    
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        
        if i < 10:
            # First 10 are wrong predictions
            ax.set_title(f"pred:{y_pred[idx]} real:{y_true[idx]}", color='red', fontsize=10)
        else:
            # Remaining 30 are correct predictions
            ax.set_title(f"{y_pred[idx]}", color='green', fontsize=10)
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('plots/prediction_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    
    _ , _ , X_test, y_test = load_mnist()
    model = load_model()
    
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    
    final_test_acc = np.sum(preds == y_test) * 100 / len(y_test)
    print(f'Test accuracy verified: {final_test_acc:.2f}%')
    
    plot_training_curves('training_history.npz', final_test_acc)
    plot_confusion_matrix(y_test, preds)
    plot_prediction_grid(X_test, y_test, preds)
    
    print('Plots successfully generated and saved in the plots/ folder.')
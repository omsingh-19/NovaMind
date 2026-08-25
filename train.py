import numpy as np
from src.losses import CrossEntropyLoss
from src.network import MLP
from src.utils import load_mnist , one_hot_encode
from config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_1,
    HIDDEN_2,
    INPUT_DIM,
    LEARNING_RATE,
    OUTPUT_DIM,
    SEED,
)

np.random.seed(SEED)

X_train, y_train, X_test, y_test = load_mnist()
y_train_oh = one_hot_encode(y_train)
   
N = X_train.shape[0]

model = MLP(INPUT_DIM,HIDDEN_1,HIDDEN_2,OUTPUT_DIM)
criterion = CrossEntropyLoss()

for epoch in range(EPOCHS):
    indices = np.random.permutation(N)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = y_train_oh[indices]

    running_loss = 0.0
    correct = 0

    # Mini-Batch loop
    for start_idx in range(0,N,BATCH_SIZE):

        end_idx = start_idx + BATCH_SIZE
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = Y_train_shuffled[start_idx:end_idx]

        pred = model.forward(X_batch)
        loss = criterion.forward(pred,y_batch)
        running_loss += loss*len(X_batch)
        correct += np.sum(np.argmax(pred,axis=1)==np.argmax(y_batch,axis=1))

        model.backward(y_batch)
        model.update(LEARNING_RATE)

    epoch_loss = running_loss / N
    epoch_acc = (correct / N) * 100
    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%"
    )

# --- Save weights after training ---
print('\n Training Completed. Saving Weights')
weights_dict = {}
idx=1
for layer in model.layers:
    if hasattr(layer,'W'):
        weights_dict[f'W{idx}']=layer.W
        weights_dict[f'b{idx}']=layer.b
        idx+=1
np.savez('model_weights.npz',**weights_dict)
print('Weights saved successfully to model_weights.npz')


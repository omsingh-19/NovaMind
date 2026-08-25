import numpy as np
from src.utils import load_mnist
from src.network import MLP
from config import (
    INPUT_DIM,
    HIDDEN_1,
    HIDDEN_2,
    OUTPUT_DIM
)

_ , _ , X_test , y_test = load_mnist()
model = MLP(INPUT_DIM,HIDDEN_1,HIDDEN_2,OUTPUT_DIM)

data = np.load('model_weights.npz')
dense_list = [layer for layer in model.layers if hasattr(layer,'W')]
for idx,layer in enumerate(dense_list,start=1):
    layer.W = data[f'W{idx}']
    layer.b = data[f'b{idx}']

# forward pass
probs = model.forward(X_test)
preds = np.argmax(probs,axis=1)
correct_preds = np.sum(preds==y_test)

accuracy = correct_preds*100/len(y_test)
print(f'The test accuracy is {accuracy:.2f}%')
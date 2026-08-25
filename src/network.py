from src.layers import Dense
from src.activations import ReLU , Softmax

class Network:

    def __init__(self,layers):
        self.layers = layers

    def forward(self,X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self,y_true):
        grad = y_true
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def update(self,lr):
        for layer in self.layers:
            if hasattr(layer,'W'):
                layer.W -= lr*layer.dW
                layer.b -= lr*layer.db

class MLP(Network):

    def __init__(self,*args):
        layers = []
        sizes = args
        for i in range(len(sizes)-1):
            layers.append(Dense(sizes[i],sizes[i+1]))
            if i == len(sizes)-2:
                layers.append(Softmax())
            else:
                layers.append(ReLU())
        super().__init__(layers)
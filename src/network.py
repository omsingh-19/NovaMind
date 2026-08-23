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

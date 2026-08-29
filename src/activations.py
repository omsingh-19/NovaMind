import numpy as np

class ReLU:

    def __init__(self):
        self.input = None

    def forward(self,z:np.ndarray)-> np.ndarray:
        self.input = z
        return np.maximum(0,z)

    def backward(self,d_out:np.ndarray)-> np.ndarray:
        return d_out * (self.input>0)

class Softmax:

    def __init__(self):
        self.prediction = None

    def forward(self,z:np.ndarray)-> np.ndarray:
        shift_z = z - np.max(z,keepdims=True,axis=-1)
        exp_z = np.exp(shift_z)
        self.prediction = exp_z/np.sum(exp_z,keepdims=True,axis=-1)
        return self.prediction

    def backward(self,y_true:np.ndarray)-> np.ndarray:

        batch_size = y_true.shape[0] if y_true.ndim>1 else 1

        return (self.prediction - y_true)/batch_size

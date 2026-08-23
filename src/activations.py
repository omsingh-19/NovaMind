import numpy as np

class ReLU:

    def __init__(self):
        self.input = None

    def forward(self,z:np.ndarray)-> np.ndarray:
        """Forward Pass: max(0,z)"""
        self.input = z
        return np.maximum(0,z)

    def backward(self,d_out:np.ndarray)-> np.ndarray:
        """Backward Pass: propagates gradient only when input > 0"""
        return d_out * (self.input>0)

class Softmax:

    def __init__(self):
        self.prediction = None

    def forward(self,z:np.ndarray)-> np.ndarray:
        """Forward pass with numerical stability trick (subtract max along class axis)

        Supports both 1D (single sample) and 2D (batch_size, num_classes) inputs.
        """
        shift_z = z - np.max(z,keepdims=True,axis=-1)
        exp_z = np.exp(shift_z)
        self.prediction = exp_z/np.sum(exp_z,keepdims=True,axis=-1)
        return self.prediction

    def backward(self,y_true:np.ndarray)-> np.ndarray:
        """Combined Softmax + Cross-Entropy gradient: dL/dz = predictions - y_true

        Normalizes by batch size to prevent gradients from exploding with larger batches.
        """
        batch_size = y_true.shape[0] if y_true.ndim>1 else 1

        return (self.prediction - y_true)/batch_size

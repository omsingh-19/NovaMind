import numpy as np

class Dense:

    def __init__(self,input_dim,output_dim):
        self.W = np.random.randn(input_dim,output_dim)*0.01
        self.b = np.zeros((1,output_dim))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self,X:np.ndarray)-> np.ndarray:
        """Computes z = X @ W + b

        - Cache input X for backward pass
        - Return the linear combination z
        """
        self.X = X
        z = X@self.W +self.b
        return z

    def backward(self,d_out:np.ndarray)-> np.ndarray:
        """Computes dW, db, and d_input given incoming d_out (dL/dz)

        d_out shape: (batch_size, output_dim)
        """
        self.dW = self.X.T @ d_out
        self.db = d_out.sum(keepdims=True,axis=0)
        d_input = d_out @ self.W.T
        return d_input

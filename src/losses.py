import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self,prediction,y_true):
        epsilon = 1e-15
        prediction = np.clip(prediction,epsilon,1-epsilon)
        log_pred = np.log(prediction)
        log_prob = log_pred*y_true
        sample_losses = np.sum(log_prob,axis=1)
        loss = -np.mean(sample_losses)
        return loss


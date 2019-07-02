from .layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, drop_rate, is_test=False, **kwargs):
        self.drop_rate = drop_rate
        self.is_test = is_test

    def forward(self, x):
        if self.is_test:
            return x
        else:
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            return (x * self.mask) / (1 - self.drop_rate)

    def backward(self, eta):
        return eta if self.is_test else eta * self.mask
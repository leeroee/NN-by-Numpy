from .layer import Layer
from ..parameter import Parameter
import numpy as np


class Linear(Layer):
    def __init__(self, shape, requires_grad=True, bias=True, **kwargs):
        '''
        shape = (in_size, out_size)
        '''
        W = np.random.randn(*shape) * (2 / shape[0]**0.5)
        self.W = Parameter(W, requires_grad)
        self.b = Parameter(np.zeros(shape[-1]), requires_grad) if bias else None
        self.require_grad = requires_grad

    def forward(self, x):
        if self.require_grad: self.x = x
        out = np.dot(x, self.W.data)
        if self.b is not None: out = out + self.b.data
        return out

    def backward(self, eta):
        if self.require_grad:
            batch_size = eta.shape[0]
            self.W.grad = np.dot(self.x.T, eta) / batch_size
            if self.b is not None: self.b.grad = np.sum(eta, axis=0) / batch_size
        return np.dot(eta, self.W.T)
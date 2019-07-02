from .layer import Layer


class MaxPooling(Layer):
    def __init__(self, size, **kwargs):
        self.size = size

    def forward(self, x):
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        out = out.max(axis=(2, 4))
        self.index = out.repeat(self.size, axis=1).repeat(self.size, axis=2) == x
        return out

    def backward(self, eta):
        return eta.repeat(self.size, axis=1).repeat(self.size, axis=2) * self.index
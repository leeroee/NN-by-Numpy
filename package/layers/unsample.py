from .layer import Layer


class Unsample(Layer):
    def __init__(self, scale, **kwargs):
        self.sacle = scale

    def forward(self, x):
        # x.shape = N, H, W, C
        return x.repeat(self.sacle, axis=1).repeat(self.sacle, axis=2)

    def backward(self, eta):
        N, H, W, C = eta.shape
        return eta.reshape(N, H//self.sacle, self.sacle, W//self.sacle, self.sacle, C).sum(axis=(2,4))
from .layer import Layer

# 平均池化，用的很少，参考Maxpooling
class MeanPooling(Layer):
    def __init__(self, size, **kwargs):
        self.size = size

    def forward(self, x):
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        return out.mean(axis=(2, 4))

    def backward(self, eta):
        return (eta / self.size**2).repeat(self.size, axis=1).repeat(self.size, axis=2)
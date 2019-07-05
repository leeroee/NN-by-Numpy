from .layer import Layer

# 在常规卷积网络中因为有线性层，所以需要将数据进行变形
class Transform(Layer):
    def __init__(self, input_shape, output_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return x.reshape(self.output_shape)

    def backward(self, eta):
        return eta.reshape(self.input_shape)
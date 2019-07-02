from .layer import Layer


class Transform(Layer):
    def __init__(self, input_shape, output_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return x.reshape(self.output_shape)

    def backward(self, eta):
        return eta.reshape(self.input_shape)
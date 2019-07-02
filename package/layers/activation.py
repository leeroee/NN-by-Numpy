from .layer import Layer
import numpy as np


class Relu(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x < 0] = 0
        return eta


class Softmax(Layer):
    def forward(self, x):
        v = np.exp(x - x.max(axis=-1, keepdims=True))    # X.shape = batch, N_classes
        self.a = v / v.sum(axis=-1, keepdims=True)      
        return self.a
    
    def backward(self, y):
        '''
        一般Softmax的反向传播和CrossEntropyLoss的放在一起
        '''
        pass


class Sigmoid(Layer):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, eta):
        return self.y * (1 - self.y)


class Tanh(Layer):
    def forward(self, x):
        self.y = 1 - ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))**2
        return self.y

    def backward(self, eta):
        return 1 - self.y**2
from .layer import Layer
import numpy as np


class Relu(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x<=0] = 0
        return eta


class Softmax(Layer):
    def forward(self, x):
        '''
        x.shape = (N, C)
        接收批量的输入，每个输入是一维向量
        计算公式为：
        a_{ij}=\frac{e^{x_{ij}}}{\sum_{j}^{C} e^{x_{ij}}}
        '''
        v = np.exp(x - x.max(axis=-1, keepdims=True))    
        return v / v.sum(axis=-1, keepdims=True)
    
    def backward(self, y):
        # 一般Softmax的反向传播和CrossEntropyLoss的放在一起
        pass


class Sigmoid(Layer):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', self.y, 1 - self.y, eta, optimize=True)


class Tanh(Layer):
    def forward(self, x):
        ex = np.exp(x)
        esx = np.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', 1 - self.y, 1 + self.y, eta, optimize=True)
from .layer import Layer
from ..parameter import Parameter
import numpy as np


class Linear(Layer):
    def __init__(self, shape, requires_grad=True, bias=True, **kwargs):
        '''
        shape: (in_size, out_size)
        requires_grad: 是否在反向传播中计算权重梯度
        bias: 是否设置偏置
        '''
        W = np.random.randn(*shape) * (2 / shape[0]**0.5)
        self.W = Parameter(W, requires_grad)
        self.b = Parameter(np.zeros(shape[-1]), requires_grad) if bias else None
        self.require_grad = requires_grad

    def forward(self, x):
        if self.require_grad: self.x = x
        # 公式：a_{ik}=\sum_{j}^{C} x_{ij} w_{jk}
        a = np.dot(x, self.W.data)
        if self.b is not None: a += self.b.data
        return a

    def backward(self, eta):
        # 在反向计算中矩阵乘法涉及转置，einsum比dot稍好一点点
        if self.require_grad:
            batch_size = eta.shape[0]
            # 公式：dW_{ik}=\frac {1}{N} \sum_{j}^{C} x_{ji} da_{jk}
            self.W.grad = np.einsum('ji,jk->ik', self.x, eta) / batch_size
            # 公式：db_{*}=\frac {1}{N} \sum_{i}^{N} da_{i*}
            if self.b is not None: self.b.grad = np.einsum('i...->...', eta, optimize=True) / batch_size
        # 公式：dz_{ik}=\sum_{j}^{C} da_{ij} w_{kj}
        return np.einsum('ij,kj->ik', eta, self.W.data, optimize=True)
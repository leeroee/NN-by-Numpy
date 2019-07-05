from .layer import Layer
import numpy as np

# 使用边界元素的值对数据进行填充，填充只发生在宽(W)高(H)两个维度上
class EdgePad(Layer):
    def __init__(self, pad_width, **kwargs):
        # pad_width: 填充的宽度
        self.top = self.bottom = self.left = self.right = pad_width
    
    def forward(self, x):
        return np.pad(x, ((0, 0), (self.top, self.bottom), (self.left, self.right), (0, 0)), 'edge')

    def backward(self, eta):
        eta[:, self.top, :, :] += eta[:, :self.top, :, :].sum(axis=1)
        eta[:, -self.bottom-1, :, :] += eta[:, -self.bottom:, :, :].sum(axis=1)
        eta[:, :, self.left, :] += eta[:, :, :self.left, :].sum(axis=2)
        eta[:, :, -self.right-1, :] += eta[:, :, -self.right:, :].sum(axis=2)
        return eta[:,self.top:-self.bottom, self.left:-self.right,:]
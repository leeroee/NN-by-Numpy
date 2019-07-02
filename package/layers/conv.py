from .layer import Layer
from ..parameter import Parameter
from functools import reduce
import numpy as np


def split_by_strides(X, kh, kw, s=1):
    '''
    将X按s的步长划分为(kh, kw)大小的子矩阵,当不能被步长整除时，
    不会发生越界，但是会有一部分信息数据不会被使用
    '''
    N, H, W, C = X.shape
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1
    shape = (N, oh, ow, kh, kw, C)
    strides = (X.strides[0], X.strides[1] * s, X.strides[2] * s, *X.strides[1:])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


class Conv(Layer):
    def __init__(self, shape, method='VALID', stride=1, require_grad=True, bias=True, **kwargs):
        '''
        shape = (out_channel, kernel_size, kernel_size, in_channel)
        '''
        W = np.random.randn(*shape) * (2 / reduce(lambda x, y: x * y, shape[1:])**0.5)
        self.W = Parameter(W, require_grad)
        self.b = Parameter(np.zeros(shape[0]), require_grad) if bias else None
        self.method = method
        self.s = stride
        self.kn = shape[0]
        self.ksize = shape[1]
        self.require_grad = require_grad

    def padding(self, x, forward=True):
        p = self.ksize // 2 if self.method == 'SAME' else self.ksize - 1
        if forward:
            return x if self.method == 'VALID' else np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        else:
            return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

    def forward(self, x):
        x = self.padding(x)                                                     # x.shape = N, iH, iW, iC
        if self.s > 1:
            self.oh = x.shape[1] - self.ksize + 1
            self.ow = x.shape[2] - self.ksize + 1
        self.x_split = split_by_strides(x, self.ksize, self.ksize, self.s)      # x_split.shape = N, oH, oW, kh, kw, iC
        a = np.tensordot(self.x_split, self.W.data, axes=[(3, 4, 5), (1, 2, 3)])# return.shape = N, oH, oW, oC
        if self.b is not None:
            a = a + self.b.data
        return a

    def backward(self, eta):
        if self.require_grad:
            batch_size = eta.shape[0]                                               # eta.shape = N, oH, oW, oC
            self.W.grad = np.tensordot(eta, self.x_split, [(0, 1, 2), (0, 1, 2)]) / batch_size  # dW.shape = oC, kh, kw, iC
            if self.b is not None:
                self.b.grad = np.reshape(eta, [eta.shape[0], -1, self.kn]).sum(axis=(0, 1)) / batch_size

        if self.s > 1:
            temp = np.zeros((eta.shape[0], self.oh, self.ow, eta.shape[3]))
            temp[:, ::self.s, ::self.s, :] = eta
            eta = temp
        eta_pad = self.padding(eta, False)
        W_rot180 = self.W.data[:, ::-1, ::-1, :].swapaxes(0, 3)                 # W_rot180.shape = iC, kh, kw, oC
        eta_split = split_by_strides(eta_pad, self.ksize, self.ksize, self.s)   # eta_split.shape = N, iH, iW, kh, kw, oC
        return np.tensordot(eta_split, W_rot180, axes=[(3, 4, 5), (1, 2, 3)])   # return.shape = N, iH, iW, iC
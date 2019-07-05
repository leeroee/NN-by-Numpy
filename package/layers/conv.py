from .layer import Layer
from ..parameter import Parameter
from functools import reduce
import numpy as np


# def split_by_strides(X, kh, kw, s=1):
#     '''
#     将数据按s的步长划分为(kh, kw)大小的子矩阵,当不能被步长整除时，
#     不会发生越界，但是会有一部分信息数据不会被使用
#     '''
#     N, H, W, C = X.shape
#     oh = (H - kh) // s + 1
#     ow = (W - kw) // s + 1
#     shape = (N, oh, ow, kh, kw, C)
#     strides = (X.strides[0], X.strides[1] * s, X.strides[2] * s, *X.strides[1:])
#     return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


class Conv(Layer):
    def __init__(self, shape, method='VALID', stride=1, requires_grad=True, bias=True, **kwargs):
        '''
        shape: (out_channel, kernel_size, kernel_size, in_channel)
        method: 填充方式可选值有{'VALID', 'SAME'}
        stride: 卷积步长
        requires_grad: 是否在反向传播中计算权重梯度
        bias: 是否设置偏置
        '''
        W = np.random.randn(*shape) * (2 / reduce(lambda x, y: x * y, shape[1:])**0.5)
        self.W = Parameter(W, requires_grad)
        self.b = Parameter(np.zeros(shape[0]), requires_grad) if bias else None
        self.method = method
        self.s = stride
        self.kn = shape[0]      # 卷积核数量，等于输出通道数
        self.ksize = shape[1]
        self.require_grad = requires_grad
        self.first_forward = True
        self.first_backward = True

    def padding(self, x, forward=True):
        # 根据填充方式以及处于前向过程还是反向过程自动对数据进行填充
        p = self.ksize // 2 if self.method == 'SAME' else self.ksize - 1
        if forward:
            return x if self.method == 'VALID' else np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        else:
            return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

    def split_by_strides(self, x):
        # 将数据按卷积步长划分为与卷积核相同大小的子集,当不能被步长整除时，不会发生越界，但是会有一部分信息数据不会被使用
        N, H, W, C = x.shape
        oh = (H - self.ksize) // self.s + 1
        ow = (W - self.ksize) // self.s + 1
        shape = (N, oh, ow, self.ksize, self.ksize, C)
        strides = (x.strides[0], x.strides[1] * self.s, x.strides[2] * self.s, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def forward(self, x):
        x = self.padding(x)
        if self.s > 1:
            # 如果卷积的步长大于1，需要计算出步长为1时输出数据的宽、高，以便在反向过程中对传入的梯度进行还原
            self.oh = x.shape[1] - self.ksize + 1
            self.ow = x.shape[2] - self.ksize + 1
        self.x_split = self.split_by_strides(x)
        if self.first_forward:
            # 在第一次训练时计算优化路径
            self.first_forward = False
            self.forward_path = np.einsum_path('ijk...,o...->ijko', self.x_split, self.W.data, optimize='greedy')[0]
        a = np.einsum('ijk...,o...->ijko', self.x_split, self.W.data, optimize=self.forward_path)
        return a if self.b is None else a + self.b.data

    def backward(self, eta):
        if self.require_grad:
            if self.first_backward:
                # 在第一次训练时计算优化路径
                self.W_grad_path = np.einsum_path('...i,...jkl->ijkl', eta, self.x_split, optimize='greedy')[0]
                self.b_grad_path = np.einsum_path('...i->i', eta, optimize='greedy')[0]
            self.W.grad = np.einsum('...i,...jkl->ijkl', eta, self.x_split, optimize=self.W_grad_path) / eta.shape[0]
            if self.b is not None:
                self.b.grad = np.einsum('...i->i', eta, optimize=self.b_grad_path) / eta.shape[0]

        if self.s > 1:
            temp = np.zeros((eta.shape[0], self.oh, self.ow, eta.shape[3]))
            temp[:, ::self.s, ::self.s, :] = eta
            eta = temp

        if self.first_backward:
            # 在第一次训练时计算优化路径
            self.first_backward = False
            self.backward_path = np.einsum_path('ijklmn,nlmo->ijko', self.split_by_strides(self.padding(eta, False)),
                self.W.data[:, ::-1, ::-1, :], optimize='greedy')[0]
        return np.einsum('ijklmn,nlmo->ijko', self.split_by_strides(self.padding(eta, False)), 
            self.W.data[:, ::-1, ::-1, :], optimize=self.backward_path)
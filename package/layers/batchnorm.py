from .layer import Layer
from ..parameter import Parameter
import numpy as np


class BatchNorm(Layer):
    def __init__(self, shape, requires_grad=True, affine=True, is_test=False, **kwargs):
        if affine:
            # 针对输入的归一化不需要仿射变换的参数
            self.gamma = Parameter(np.random.uniform(0.9, 1.1, shape), requires_grad, True)
            self.beta = Parameter(np.random.uniform(-0.1, 0.1, shape), requires_grad, True)
            self.requires_grad = requires_grad
        self.eps = 1e-8
        self.affine = affine
        self.is_test = is_test
        self.coe = 0.02
        self.overall_var = Parameter(np.zeros(shape), False)
        self.overall_ave = Parameter(np.zeros(shape), False)

    def forward(self, x):
        if self.is_test:
            # 进行测试时使用估计的训练集的整体方差和均值进行归一化
            sample_ave = self.overall_ave.data
            sample_std = np.sqrt(self.overall_var.data)
        else:
            # 进行训练时使用样本的均值和方差对训练集整体的均值和方差进行估计（使用加权平均的方法）
            sample_ave = x.mean(axis=0)
            sample_var = x.var(axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            self.overall_ave.data = (1 - self.coe) * self.overall_ave.data + self.coe * sample_ave
            self.overall_var.data = (1 - self.coe) * self.overall_var.data + self.coe * sample_var
        return (x - sample_ave) / sample_std if not self.affine else self.forward_internal(x - sample_ave, sample_std)

    def backward(self, eta):
        if not self.affine: return              # 如果是针对输入层做归一化就不存在向上传播梯度了
        self.beta.grad = eta.mean(axis=0)
        self.gamma.grad = (eta * self.normalized).mean(axis=0)
        return self.gamma_s * (eta - self.normalized * self.gamma.grad - self.beta.grad)

    def forward_internal(self, sample_diff, sample_std):
        '''
        如果是在网络内部使用Batch Norm需要进一步进行仿射变化，如果是对输入进行归一化就不用了
        '''
        self.normalized = sample_diff / sample_std
        self.gamma_s = self.gamma.data / sample_std
        return self.gamma.data * self.normalized + self.beta.data

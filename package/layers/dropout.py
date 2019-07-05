from .layer import Layer
import numpy as np

# 随机失活
class Dropout(Layer):
    def __init__(self, drop_rate, is_test=False, **kwargs):
        '''
        drop_rate: 失活概率
        is_test: 是否为测试状态
        '''
        self.drop_rate = drop_rate
        self.fix_value = 1 / (1 - drop_rate)    # 修正值，使forward输出整体期望保持不变
        self.is_test = is_test
        self.first_forward = True
        self.first_backward = True

    def forward(self, x):
        if self.is_test:
            # 如果是测试，直接返回
            return x
        else:
            # 如果是训练，按概率将输出置为0
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            if self.first_forward:
                # 在第一次训练时计算优化路径
                self.first_forward = False
                self.forward_path = np.einsum_path('...,...,->...', x, self.mask, self.fix_value, optimize='greedy')[0]
            return np.einsum('...,...,->...', x, self.mask, self.fix_value, optimize=self.first_forward)

    def backward(self, eta):
        if self.is_test:
            return eta
        else:
            if self.first_backward:
                # 在第一次训练时计算优化路径
                self.first_backward = False
                self.backward_path = np.einsum_path('...,...->...', eta, self.mask, optimize='greedy')[0]
            return np.einsum('...,...->...', eta, self.mask, optimize=self.backward_path)
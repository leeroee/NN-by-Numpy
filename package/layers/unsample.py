from .layer import Layer
import numpy as np

# 上采样层，对数据进行复制扩展
class Unsample(Layer):
    def __init__(self, scale, **kwargs):
        # scale: 扩展规模
        self.sacle = scale
        self.first_backward = True

    def forward(self, x):
        return x.repeat(self.sacle, axis=1).repeat(self.sacle, axis=2)

    def backward(self, eta):
        N, H, W, C = eta.shape
        eta = eta.reshape(N, H//self.sacle, self.sacle, W//self.sacle, self.sacle, C)
        if self.first_backward:
            # 在第一次训练时计算优化路径
            self.first_backward = False
            self.backward_path = np.einsum_path('ijklmn->ijln', eta, optimize='greedy')[0]
        return np.einsum('ijklmn->ijln', eta, optimize=True)
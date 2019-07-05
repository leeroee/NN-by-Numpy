from .layer import Layer

# 最大池化
class MaxPooling(Layer):
    def __init__(self, size, **kwargs):
        '''
        size: Pooling的窗口大小，因为在使用中窗口大小与步长基本一致，所以简化为一个参数
        '''
        self.size = size

    def forward(self, x):
        # 首先将输入按照窗口大小划分为若干个子集
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        # 取每个子集的最大值
        out = out.max(axis=(2, 4))
        # 记录每个窗口中不是最大值的位置
        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        return out

    def backward(self, eta):
        # 将上一层传入的梯度进行复制，使其shape扩充到forward中输入的大小
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        # 将不是最大值的位置的梯度置为0
        eta[self.mask] = 0
        return eta
from .layer import Layer
import numpy as np

# 以边界的元素为对称轴，对数据进行镜像填充，填充只发生在宽(W)高(H)两个维度上
class ReflectionPad(Layer):
    def __init__(self, pad_width, **kwargs):
        # pad_width: 填充的宽度
        self.top = self.bottom = self.left = self.right = pad_width

    def forward(self, x):
        return np.pad(x, ((0, 0), (self.top, self.bottom), (self.left, self.right), (0, 0)), 'reflect')

    def backward(self, eta):
        eta[:,self.top+1:2*self.top+1,:,:] += eta[:,self.top-1::-1,:,:]
        eta[:,-1-2*self.bottom:-1-self.bottom,:,:] += eta[:,:-self.bottom-1:-1,:,:]
        eta[:,:,self.left+1:2*self.left+1,:] += eta[:,:,self.left-1::-1,:]
        eta[:,:,-1-2*self.right:-1-self.right,:] += eta[:,:,:-self.right-1:-1,:]
        return eta[:,self.top:-self.bottom, self.left:-self.right,:]
from .layer import Layer
import numpy as np


class ReflectionPad(Layer):
    def __init__(self, pad_width, **kwargs):
        self.top = self.bottom = self.left = self.right = pad_width

    def forward(self, x):
        return np.pad(x, ((0, 0), (self.top, self.bottom), (self.left, self.right), (0, 0)), 'reflect')

    def backward(self, eta):
        eta[:,self.top+1:2*self.top+1,:,:] += eta[:,self.top-1::-1,:,:]
        eta[:,-1-2*self.bottom:-1-self.bottom,:,:] += eta[:,:-self.bottom-1:-1,:,:]
        eta[:,:,self.left+1:2*self.left+1,:] += eta[:,:,self.left-1::-1,:]
        eta[:,:,-1-2*self.right:-1-self.right,:] += eta[:,:,:-self.right-1:-1,:]
        return eta[:,self.top:-self.bottom, self.left:-self.right,:]
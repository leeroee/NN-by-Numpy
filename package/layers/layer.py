from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    '''
    作为所有层的基类，如果自定义新的层应该从此类继承并重写下面两个方法
    '''
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass
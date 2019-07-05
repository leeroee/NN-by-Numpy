import numpy as np

# 均方误差损失函数
class MSELoss(object):
    def gradient(self):
        # 返回损失关于输出的梯度
        # \frac{\partial L}{\partial a_{ij}}=\frac{a_{ij}-y_{ij}}{C}
        return self.u / self.u.shape[1]
    
    def __call__(self, a, y):
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        return: 该批样本的平均损失

        输出与真值的shape是一样的，并且都是批量的，单个输出与真值是一维向量
        a.shape = y.shape = (N, C)      N是该批样本的数量，C是单个样本最终输出向量的长度
        '''
        # u_{ij} = a_{ij} - y_{ij}
        self.u = a - y
        # 批样本的整体损失
        # L_{i}=\frac{1}{C} \sum_{j}^{C}\left(a_{ij}-y_{ij}\right)^{2}=\frac{1}{C} \sum_{j}^{C} u_{ij} u_{ij}
        # 样本的平均损失
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=\frac{1}{NC} \sum_{i}^{N} \sum_{j}^{C} u_{ij} u_{ij}
        return np.einsum('ij,ij->', self.u, self.u, optimize=True) / self.u.size
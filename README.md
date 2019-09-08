# NN-by-Numpy
使用Numpy实现的一个小型神经网络框架，包括DNN、CNN中的各种Layer，诸如线形成、卷积层、池化层，ReLu、Sigmoid、Softmax等激活函数以及均方误差、交叉熵等损失函数

代码中有些使用了einsum函数，该函数的使用方法见：[爱因斯坦求和约定](https://zhuanlan.zhihu.com/p/71639781)

然后是各个模块的具体原理，参见我的知乎文章
- [Numpy实现神经网络框架(1)](https://zhuanlan.zhihu.com/p/67716530)
- [Numpy实现神经网络框架(2)——梯度下降、反向传播](https://zhuanlan.zhihu.com/p/74233026)
- [Numpy实现神经网络框架(3)——线性层反向传播推导及实现](https://zhuanlan.zhihu.com/p/67854272)
  - 扩展：[Softmax与交叉熵损失的实现及求导](https://zhuanlan.zhihu.com/p/67759205)
- [Numpy实现神经网络框架(4)——MNIST手写数字识别](https://zhuanlan.zhihu.com/p/67972253)
- [Numpy实现神经网络框架(5)——梯度下降优化算法](https://zhuanlan.zhihu.com/p/68093219)
  - 扩展：[从SGD到NadaMax，十种优化算法原理及实现](https://zhuanlan.zhihu.com/p/81020717)
  
  
- [Numpy实现神经网络框架(6)——归一化浅析](https://zhuanlan.zhihu.com/p/68211390)
- [Numpy实现神经网络框架(7)——Batch Normalization](https://zhuanlan.zhihu.com/p/68685625)
  - 扩展：[各种归一化实现及梯度推导——Batch、Layer、Instance、Switchable Norm](https://zhuanlan.zhihu.com/p/74907399)
- [Numpy实现神经网络框架(8)——卷积神经网络基础](https://zhuanlan.zhihu.com/p/69229755)
  - 扩展：[im2col方法实现卷积算法](https://zhuanlan.zhihu.com/p/63974249)
  - 扩展：[卷积核梯度计算的推导及实现](https://zhuanlan.zhihu.com/p/64248652)
  - 扩展：[卷积算法另一种高效实现，as_strided详解](https://zhuanlan.zhihu.com/p/64933417)
- [Numpy实现神经网络框架(9)——卷积层反向传播推导及实现](https://zhuanlan.zhihu.com/p/70246295)
- [Numpy实现神经网络框架(10)——CNN中的Padding、Pooling](https://zhuanlan.zhihu.com/p/70713747)
  - [ReflectionPad2d、InstanceNorm2d详解及实现](https://zhuanlan.zhihu.com/p/66989411)

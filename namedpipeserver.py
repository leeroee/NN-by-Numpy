import win32api
import win32pipe
import win32file
import numpy as np
import package
import package.optim as optim
import os


def save(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as, **dic)
    
def load(parameters, file):
    params = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]


def load_MNIST(file, transform=False):
    file = np.load(file)
    X = file['X']
    Y = file['Y']
    if transform:
        X = X.reshape(len(X), -1)
    return X, Y


def train(net, loss_fn, train_file, batch_size, optimizer, load_file, save_as, times=1, retrain=False, transform=False):
    X, Y = load_MNIST(train_file, transform)
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    
    
    PIPE_NAME = r'\\.\pipe\nnpipe'      # 管道名称
    PIPE_BUFFER_SIZE = 14               # 管道缓存大小
    
    # 创建命名管道
    named_pipe = win32pipe.CreateNamedPipe(PIPE_NAME,
                                           win32pipe.PIPE_ACCESS_DUPLEX,
                                           win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT | win32pipe.PIPE_READMODE_MESSAGE,
                                           win32pipe.PIPE_UNLIMITED_INSTANCES,
                                           PIPE_BUFFER_SIZE,
                                           PIPE_BUFFER_SIZE, 500, None)
    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:            
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size
            x = x / 255
            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output, y)
            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update()
            print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                (loop, i, batch_acc*100, batch_loss))
            

            win32pipe.ConnectNamedPipe(named_pipe, None)    # 连接管道
            msg = '%5d#%.6f' % (i, batch_loss)
            win32file.WriteFile(named_pipe, msg.encode())   # 向管道发送信息
            win32pipe.DisconnectNamedPipe(named_pipe)       # 断开管道

    if save_as is not None:
        save(net.parameters, save_as)
    
    win32api.CloseHandle(named_pipe)# 关闭命名管道

if __name__ == "__main__": 
    layers = [
        {'type': 'conv', 'shape': (8, 5, 5, 1)},
        {'type': 'relu'},
        {'type': 'maxpool', 'size': 2},
        {'type': 'conv', 'shape': (16, 5, 5, 8)},
        {'type': 'relu'},
        {'type': 'maxpool', 'size': 2},
        {'type': 'transform', 'input_shape': (-1, 4, 4, 16), 'output_shape': (-1, 256)},
        {'type': 'linear', 'shape': (256, 64)},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (64, 10)},
    ]
    loss_fn = package.CrossEntropyLoss()
    net = package.Net(layers)
    optimizer = optim.Adam(net.parameters, 0.01)
    train_file = './MNIST/trainset.npz'
    param_file = './MNIST/param.npz'
    batch_size = 128
    train(net, loss_fn, train_file, batch_size, optimizer, param_file, param_file, times=1, retrain=True) 



import numpy as np
import package
import package.optim as optim
import os
import sys


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


def train(net, loss_fn, train_file, batch_size, optimizer, load_file, save_as, times=1, retrain=False):
    X, Y = load_MNIST(train_file, transform=True)
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output, y)
            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update()
            print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                (loop, i, batch_acc*100, batch_loss))
        pass
    if save_as is not None: save(net.parameters, save_as)
    

if __name__ == "__main__": 
    layers = [
        {'type': 'batchnorm', 'shape': 784, 'requires_grad': False, 'affine': False},
        {'type': 'linear', 'shape': (784, 400)},
        {'type': 'batchnorm', 'shape': 400},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (400, 100)},
        {'type': 'batchnorm', 'shape': 100},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (100, 10)}
    ]
    loss_fn = package.CrossEntropyLoss()
    net = package.Net(layers)
    lr = 0.001
    batch_size = 128
    optimizer = optim.Adam(net.parameters, lr)
    train_file = './MNIST/trainset.npz'
    param_file = './MNIST/param.npz'
    train(net, loss_fn, train_file, batch_size, optimizer, param_file, param_file, times=1, retrain=True)

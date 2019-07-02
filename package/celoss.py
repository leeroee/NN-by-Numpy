from .layers.activation import Softmax
import numpy as np


class CrossEntropyLoss(object):
    def __init__(self):
        self.classifier = Softmax()

    def gradient(self):
        return self.a - self.y

    def __call__(self, output, target, requires_acc=True):
        self.a = self.classifier.forward(output)
        self.y = target
        loss = np.log((self.a * self.y).sum(axis=-1)).mean()
        if requires_acc:
            acc = np.argmax(self.a, axis=-1) == np.argmax(self.y, axis=-1)
            return acc.mean(), -loss
        return -loss
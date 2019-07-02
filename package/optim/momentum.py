import numpy as np


class Momentum(object):
    def __init__(self, parameters, lr, decay=0, beta=0.9):
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.beta = beta
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accmulated_grads = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        for p, grad in zip(self.parameters, self.accmulated_grads):
            if self.decay_rate < 1 and not p.skip_decay: p.data *= self.decay_rate
            np.copyto(grad, self.beta * grad + (1 - self.beta) * p.grad)
            p.data -= self.lr * grad
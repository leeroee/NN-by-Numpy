import numpy as np


class RMSprpo(object):
    def __init__(self, parameters, lr, decay=0, beta=0.98, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accumulated_grads = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        for p, grad in zip(self.parameters, self.accumulated_grads):
            if self.decay_rate < 1 and not p.skip_decay: p.data *= self.decay_rate
            np.copyto(grad, self.beta * grad + (1 - self.beta) * np.power(p.grad, 2))
            p.data -= self.lr * p.grad / (np.sqrt(grad) + self.eps)
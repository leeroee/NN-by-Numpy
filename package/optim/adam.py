import numpy as np


class Adam(object):
    def __init__(self, parameters, lr, decay=0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.accumulated_beta1 = 1
        self.accumulated_beta2 = 1
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.eps = eps
        self.parameters = [p for p in parameters if p.requires_grad]
        self.accumulated_grad_mom = [np.zeros(p.data.shape) for p in self.parameters]
        self.accumulated_grad_rms = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        self.accumulated_beta1 *= self.beta1
        self.accumulated_beta2 *= self.beta2
        lr = self.lr * ((1 - self.accumulated_beta2)**0.5) / (1 - self.accumulated_beta1)
        for p, grad_mom, grad_rms in zip(self.parameters, self.accumulated_grad_mom, self.accumulated_grad_rms):
            if self.decay_rate < 1 and not p.skip_decay: p.data *= self.decay_rate
            np.copyto(grad_mom, self.beta1 * grad_mom + (1 - self.beta1) * p.grad)
            np.copyto(grad_rms, self.beta2 * grad_rms + (1 - self.beta2) * np.power(p.grad, 2))
            p.data -= lr * grad_mom / (np.sqrt(grad_rms) + self.eps)
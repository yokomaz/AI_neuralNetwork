import numpy as np
import math

def one_hot_encode(labels, num_classes=10):
    one_hot_matrix = np.zeros((labels.size, num_classes), dtype=int)
    flat_labels = labels.flatten()
    one_hot_matrix[np.arange(flat_labels.size), flat_labels] = 1
    return one_hot_matrix    

class lr_scheduler:
    def __init__(self, lr, gamma):
        self.lr = lr
        self.gamma = gamma

    def step(self):
        self.lr = self.lr * self.gamma
        return self.lr
    
    def get_lr(self):
        return self.lr

class cos_lr_scheduler:
    def __init__(self, lr_start, lr_max, lr_min, warm_up_epoch, max_epoch):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_epoch = warm_up_epoch
        self.max_epoch = max_epoch
        self.current_epoch = 0
        self.lr = lr_start

    def step(self):
        self.current_epoch += 1
        if self.current_epoch < self.warmup_epoch:
            lr = self.lr_max * (self.current_epoch / self.warmup_epoch)
            self.lr = lr
        else:
            # Cosine 退火阶段
            progress = (self.current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
            self.lr = lr
        return self.lr
    
    def get_lr(self):
        return self.lr

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr_schech = lr

    def step(self):
        lr = self.lr_schech.get_lr()
        # for param in self.params:
        self.params['input_layer_weights'] -= lr * self.params['input_layer_weights_grad']
        self.params['input_layer_bias'] -= lr * self.params['input_layer_bias_grad']
        self.params['hidden_layer_weights'] -= lr * self.params['hidden_layer_weights_grad']
        self.params['hidden_layer_bias'] -= lr * self.params['hidden_layer_bias_grad']
        self.params['output_layer_weights'] -= lr * self.params['output_layer_weights_grad']
        self.params['output_layer_bias'] -= lr * self.params['output_layer_bias_grad']
    
    def zero_grad(self):
        # for param in self.params:
        self.params['input_layer_weights_grad'] = np.zeros_like(self.params['input_layer_weights'])
        self.params['input_layer_bias_grad'] = np.zeros_like(self.params['input_layer_bias'])
        self.params['hidden_layer_weights_grad'] = np.zeros_like(self.params['hidden_layer_weights'])
        self.params['hidden_layer_bias_grad'] = np.zeros_like(self.params['hidden_layer_bias'])
        self.params['output_layer_weights_grad'] = np.zeros_like(self.params['output_layer_weights'])
        self.params['output_layer_bias_grad'] = np.zeros_like(self.params['output_layer_bias'])
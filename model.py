import numpy as np
import pickle
import os

np.random.seed(42)
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size        
        # 初始化权重和偏置
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.Z = np.dot(inputs, self.weights) + self.bias
        return self.Z
    
    def backward(self, dL_dZ, l2_lambda, learning_rate=0.01, clip_value=1.0):
        # with the assumption dL/dW = dL/dy * dy/da * da/dz * dz/dW\
        # where a is activate function, z = x*w+b, since y=a, then dy/da=1, dL/dy = deviation of loss_function
        # with loss function is cross entropy, the deviation of loss is y_pred-y^
        dL_dw = np.dot(self.inputs.T, dL_dZ) + l2_lambda * 2 * self.weights
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        dL_dx = np.dot(dL_dZ, self.weights.T)
        # print(f"before l2 {dL_dw}")
        # exit()
        # dL_dw = dL_dw + l2_lambda * 2 * self.weights# np.sum(2 * self.weights)
        # print(f"after l2 {dL_dw}")
        dL_dw = np.clip(dL_dw, -clip_value, clip_value)
        dL_db = np.clip(dL_db, -clip_value, clip_value)
        return dL_dw, dL_db, dL_dx
        
        # 原版层反向传播
        # self.weights -= learning_rate * (dL_dw + l2_lambda * np.sum(2 * self.weights))
        # self.bias -= learning_rate * dL_db

class neuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activate='sigmoid'):
        self.input_layer = LinearLayer(input_size=input_dim, output_size=hidden_dim)
        self.hidden_layer = LinearLayer(input_size=hidden_dim, output_size=hidden_dim)
        self.output_layer = LinearLayer(input_size=hidden_dim, output_size=output_dim)
        """
        self.input_layer_weights_grads = 0
        self.input_layer_bias_grads = 0
        self.hidden_layer_weights_grads = 0
        self.hidden_layer_bias_grads = 0
        self.output_layer_weights_grads = 0
        self.output_layer_bias_grads = 0
        """
        self.activate = activate
        self.parameters = {
            "activate": self.activate, \
            "input_layer_weights": self.input_layer.weights, \
            "input_layer_weights_grad": np.zeros_like(self.input_layer.weights), \
            "input_layer_bias": self.input_layer.bias, \
            "input_layer_bias_grad": np.zeros_like(self.input_layer.bias), \
            "hidden_layer_weights": self.hidden_layer.weights, \
            "hidden_layer_weights_grad": np.zeros_like(self.hidden_layer.weights), \
            "hidden_layer_bias": self.hidden_layer.bias, \
            "hiddent_layer_bias_grad": np.zeros_like(self.hidden_layer.bias), \
            "output_layer_weights": self.output_layer.weights, \
            "output_layer_weights_grad": np.zeros_like(self.output_layer.weights), \
            "output_layer_bias": self.output_layer.bias, \
            "output_layer_bias_grad": np.zeros_like(self.output_layer.bias)}
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, output):
        z_max = np.max(output, axis=-1, keepdims=True)
        z_exp = np.exp(output - z_max)
        z_sum = np.sum(z_exp, axis=-1, keepdims=True)
        return z_exp / z_sum
    
    def save(self, model_name):
        path_to_save = './model_params'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
        self.parameters['input_layer_weights'] = self.input_layer.weights
        self.parameters['input_layer_bias'] = self.input_layer.bias
        self.parameters['hidden_layer_weights'] = self.hidden_layer.weights
        self.parameters['hidden_layer_bias'] = self.hidden_layer.bias
        self.parameters['output_layer_weights'] = self.output_layer.weights
        self.parameters['output_layer_bias'] = self.output_layer.bias
        self.parameters['activate'] = self.activate

        model_file = os.path.join(path_to_save, model_name)
        with open(model_file, 'wb') as f:
            pickle.dump(self.parameters, f)
        print(f'Model saved to path {model_file}')

    def load(self, model_name):
        with open(model_name, 'rb') as f:
            model_parameters = pickle.load(f)
        # print(model_parameters)
        self.input_layer.weights = model_parameters['input_layer_weights']
        self.input_layer.bias = model_parameters['input_layer_bias']
        self.hidden_layer.weights = model_parameters['hidden_layer_weights']
        self.hidden_layer.bias = model_parameters['hidden_layer_bias']
        self.output_layer.weights = model_parameters['output_layer_weights']
        self.output_layer.bias = model_parameters['output_layer_bias']
        self.activate = model_parameters['activate']

    def forward(self, X):
        Z1 = self.input_layer.forward(X)
        if self.activate == 'sigmoid':
            A1 = self.sigmoid(Z1)
        elif self.activate =='relu':
            A1 = self.relu(Z1)
        Z2 = self.hidden_layer.forward(A1)
        if self.activate == 'sigmoid':
            A2 = self.sigmoid(Z2)
        elif self.activate =='relu':
            A2 = self.relu(Z2)
        Z3 = self.output_layer.forward(A2)
        self.output = self.softmax(Z3)
        return self.output
    
    def backward(self, y_true, l2_lambda, learning_rate):
        # print(f"self.output {self.output.shape}, y_true {y_true.shape}")
        # exit()
        dL_dZ3 = self.output - y_true

        # backward to hidden layer
        dL_dA2 = np.dot(dL_dZ3, self.output_layer.weights.T)
        if self.activate == 'sigmoid':
            dL_dZ2 = dL_dA2 * self.sigmoid_derivative(self.hidden_layer.Z)
        elif self.activate == 'relu':
            dL_dZ2 = dL_dA2 * self.relu_derivative(self.hidden_layer.Z)
        
        dL_dA1 = np.dot(dL_dZ2, self.hidden_layer.weights.T)
        if self.activate == 'sigmoid':
            dL_dZ1 = dL_dA1 * self.sigmoid_derivative(self.input_layer.Z)
        elif self.activate == 'relu':
            dL_dZ1 = dL_dA1 * self.relu_derivative(self.input_layer.Z)
        
        # 原版反向传播函数
        # self.output_layer.backward(dL_dZ3, l2_lambda, learning_rate)
        # self.hidden_layer.backward(dL_dZ2, l2_lambda, learning_rate)
        # self.input_layer.backward(dL_dZ1, l2_lambda, learning_rate)

        # 修正版反向传播函数
        self.output_layer_weights_grads, self.output_layer_bias_grads, _ = self.output_layer.backward(dL_dZ3, l2_lambda, learning_rate)
        self.hidden_layer_weights_grads, self.hidden_layer_bias_grads, _ = self.hidden_layer.backward(dL_dZ2, l2_lambda, learning_rate)
        self.input_layer_weights_grads, self.input_layer_bias_grads, _ = self.input_layer.backward(dL_dZ1, l2_lambda, learning_rate)

        # self.parameters['input_layer_weights'] = self.input_layer.weights
        self.parameters['input_layer_weights_grad'] = self.input_layer_weights_grads
        # self.parameters['input_layer_bias'] = self.input_layer.bias
        self.parameters['input_layer_bias_grad'] = self.input_layer_bias_grads
        
        # self.parameters['hidden_layer_weights'] = self.hidden_layer.weights
        self.parameters['hidden_layer_weights_grad'] = self.hidden_layer_weights_grads
        # self.parameters['hidden_layer_bias'] = self.hidden_layer.bias
        self.parameters['hidden_layer_bias_grad'] = self.hidden_layer_bias_grads
        
        # self.parameters['output_layer_weights'] = self.output_layer.weights
        self.parameters['output_layer_weights_grad'] = self.output_layer_weights_grads
        # self.parameters['output_layer_bias'] = self.output_layer.bias
        self.parameters['output_layer_bias_grad'] = self.output_layer_bias_grads
        
        
        
import numpy as np

def relu(x):
    x[x<0] = 0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_sparse(x):
    for i in range(len(x.data)):
        x.data[i] = sigmoid(x.data[i])
    return x

def tanh(x):
    return np.tanh(x)

def tanh_sparse(x):
    for i in range(len(x.data)):
        x.data[i] = tanh(x.data[i])
    return x

activiation_functions = {'relu':relu, 'sigmoid':sigmoid, 'tanh':tanh}
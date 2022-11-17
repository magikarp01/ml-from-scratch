import numpy as np

class FullyConnectedLayer:
  # num_inputs is m
  # num_neurons is n
  def __init__(self, num_inputs, num_neurons):
    self.weights = np.random.normal(size = (num_inputs, num_neurons))
    self.biases = np.random.normal(size=(1, num_neurons))

  # x is 1xm
  # weights is mxn
  # bias is 1xn
  def forward(self, x):
    # print(x.shape)
    # print(self.weights.shape)
    # print(self.biases.shape)
    return x @ self.weights + self.biases

  def get_weights(self):
    return self.weights
  
  def get_biases(self):
    return self.biases

  # x is input from previous layer, mx1 matrix for m different neurons in prev layer
  # for layer 3, need gradient of loss w.r.t. activation of layer 3 as prev_grad
  # prev_grad is 1xn gradient for n neurons 
  def backward(self, x, prev_grad, learning_rate):
    # print("for common factor:")
    # print(prev_grad.shape)
    # print(self.weights.shape)
    # print(self.biases.shape)

    # common_factor = prev_grad * self.activation_func_deriv(x @ self.weights + self.biases) # both 1 x n matrix

    # print("for weight gradient:")
    # print(x.shape)
    # print(common_factor.shape)

    weight_gradient = np.transpose(x) @ prev_grad # (mx1) @ (1xn) returns mxn matrix
    bias_gradient = prev_grad 

    self.weights -= learning_rate * weight_gradient
    self.biases -= learning_rate * bias_gradient
    
    # returns gradient of loss w.r.t. x (activation of previous layer)
    activ_gradient = prev_grad @ np.transpose(self.weights)
    return activ_gradient
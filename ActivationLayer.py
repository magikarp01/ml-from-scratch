import numpy as np

# nx1 matrix -> nx1 matrix
def relu(x):
  return np.maximum(0, x)
  # relu_output = np.zeros_like(x)
  # for i in range(x.shape[0]):
  #   for j in range(x.shape[1]):
  #     for k in range(x.shape[2]):
  #       if x[i, j, k] >= 0:
  #         relu_output[i, j, k] = x[i, j, k]
  # return relu_output
  # return np.maximum.reduce(np.zeros_like(x), x)

# from https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
def relu_deriv(x):
  b = x > 0
  c = b.astype(int)
  return c
  # prev_grad = np.zeros_like(x)
  # for i in range(x.shape[0]):
  #   for j in range(x.shape[1]):
  #     for k in range(x.shape[2]):
  #       if x[i, j, k] > 0:
  #         prev_grad[i, j, k] = 1
  # return prev_grad

class ActivationLayer:
  def __init__(self, activ_func = relu, activ_func_deriv = relu_deriv):
    self.activ_func = activ_func
    self.activ_fun_deriv = activ_func_deriv

  def forward(self, x):
    return self.activ_func(x)
  
  # prev grad is nx1
  def backward(self, x, prev_grad, learning_rate):
    return self.activ_fun_deriv(x) * prev_grad
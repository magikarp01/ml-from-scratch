import numpy as np

def relu(x):
  relu_output = np.zeros_like(x)
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      for k in range(x.shape[2]):
        if x[i, j, k] >= 0:
          relu_output[i, j, k] = x[i, j, k]
  return relu_output
  # return np.maximum.reduce(np.zeros_like(x), x)

# assume 3 dimensions
def relu_deriv(x):
  prev_grad = np.zeros_like(x)
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      for k in range(x.shape[2]):
        if x[i, j, k] >= 0:
          prev_grad[i, j, k] = 1
  return prev_grad

class ActivationLayer:
  def __init__(self, activ_func = relu, activ_func_deriv = relu_deriv):
    self.activ_func = activ_func
    self.activ_fun_deriv = activ_func_deriv

  def forward(self, x):
    return self.activ_func(x)
  
  def backward(self, x, prev_grad, learning_rate):
    return self.activ_fun_deriv(x) * prev_grad
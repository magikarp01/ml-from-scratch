class ActivationLayer:
  def __init__(self, activ_func = relu, activ_func_deriv = relu_deriv):
    self.activ_func = activ_func
    self.activ_fun_deriv = activ_func_deriv

  def forward(self, x):
    return self.activ_func(x)
  
  def backward(self, x, prev_grad, learning_rate):
    return self.activ_fun_deriv(x) * prev_grad
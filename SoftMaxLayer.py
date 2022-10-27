class SoftmaxLayer:
  def __init__(self):
    return

  # x should be 1xn
  def softmax(self, x):
    exps = np.exp(x)
    total_exp = np.sum(exps)
    y = np.empty_like(x)
    for i in range(x.shape[1]):
      y[0, i] = exps[0, i]/total_exp

    return y

  # x is 1xn
  def forward(self, x):
    return self.softmax(x)
  
  def backward(self, x, prev_grad, learning_rate): 
    exps = np.exp(x)
    n = x.shape[1]
    total_exp = np.sum(exps)
    # print(total_exp)
    softmax_grad = np.empty((n, n))
    for i in range(n):
      for j in range(n):
        if i == j:
          softmax_grad[i, j] = exps[0,i] * (total_exp - exps[0,i]) / total_exp**2
        else:
          softmax_grad[i, j] = - exps[0,i] * exps[0,j] / total_exp**2
    # print(softmax_grad)
    # 1xn @ nxn 
    return prev_grad @ softmax_grad
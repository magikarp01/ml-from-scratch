class NeuralNetwork:
  def __init__(self, layers, verbose=False):
    self.layers = layers
    self.verbose = verbose

  def forward(self, x):
    next_x = x
    for layer in self.layers:
      next_x = layer.forward(next_x)
    return next_x

  # prev inputs is 
  def backward(self, loss_grad, prev_inputs, learning_rate):
    prev_grad = loss_grad
    for i in range(len(self.layers)-1, -1, -1):
      layer = self.layers[i]
      prev_grad = layer.backward(prev_inputs[i], prev_grad, learning_rate)
      if self.verbose:
        print(f"layer input: {layer}")
        print(f"backpropagated gradient: {prev_grad.shape}")
    
  # train on one set of data
  # returns current loss
  def train(self, x, y, learning_rate, loss_func, loss_func_deriv):
    next_x = x
    layer_inputs = []
    for layer in self.layers:
      if self.verbose:
        print(f"layer input: {layer}")
        print(next_x.shape)
      layer_inputs.append(next_x)
      next_x = layer.forward(next_x)
    output = next_x

    loss_grad = loss_func_deriv(output=output, y=y)
    self.backward(loss_grad=loss_grad, prev_inputs=layer_inputs, learning_rate=learning_rate)

    return loss_func(output=output, y=y)
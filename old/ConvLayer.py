# Convolutional, Pooling, Flattening Layers
import numpy as np
import math

# A class for a convolutional layer
class ConvLayer:
  def __init__(self, F=1, kernel_size = (3, 3, 1), same_padding = True):
    self.kernels = np.random.normal(size=(F, kernel_size[0], kernel_size[1], kernel_size[2]))
    self.same_padding = True
    self.F = F
    self.kernel_size = kernel_size
  
  def get_kernels(self):
    return self.kernels

  # array2 is smaller, filter
  # 3d is not actually needed, since z dimension of two arrays are always the same
  def convolve3d(self, array1, array2):
    shape1 = array1.shape
    shape2 = array2.shape
    # print("shapes:")
    # print(shape1)
    # print(shape2)
    x_size = shape1[0] - shape2[0] + 1
    y_size = shape1[1] - shape2[1] + 1
    z_size = shape1[2] - shape2[2] + 1
    fin_array = np.empty((x_size, y_size, z_size))
    for x in range(x_size):
      for y in range(y_size):
        for z in range(z_size):
          mat1 = array1[x:x+shape2[0], y:y+shape2[1], z:z+shape2[2]]
          fin_array[x, y, z] = np.sum(mat1 * array2)
    return fin_array

  def forward(self, feature_map):
    # will swap axes, want to iterate by F
    # don't need a feature_map.shape[2] dimension because the output of convolve3d will always have a 3rd dimension of 1
    return_map = np.empty((self.F, feature_map.shape[0], feature_map.shape[1]))

    pad_lengths = [(int(math.ceil((i-1)/2)), int(math.floor((i-1)/2))) for i in self.kernel_size]
    
    for i in range(self.F):
      if self.same_padding:
        array1 = np.pad(feature_map, pad_lengths)
      else:
        array1 = feature_map
      filter = self.kernels[i]
      # print(f'array1: {array1.shape}')
      # print(f'filter: {filter.shape}')
      return_map[i] = np.squeeze(self.convolve3d(array1, filter), 2)
    
    return_map = np.swapaxes(np.swapaxes(return_map, 0, 1), 1, 2)

    return return_map

  # for when the prev_grad has multiple channels (for each filter), more than array1's channels
  def convolve_kernel(self, array1, prev_grad):
    num_channels = self.kernel_size[2]
    kernel_gradient = np.empty_like(self.kernels)
    for i in range(self.F):
      grad_slice = prev_grad[:, :, i*num_channels:(i+1)*num_channels] 
      # kernel_gradient[:, :, i*num_channels:(i+1)*num_channels+1] = self.convolve3d(array1, grad_slice)
      kernel_gradient[i*num_channels:(i+1)*num_channels] = self.convolve3d(array1, grad_slice)
    return kernel_gradient

  # for a refresher, use https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
  def backward(self, feature_map, prev_grad, learning_rate):
    # convolve prev_grad and feature_map, prev_grad is filter
    pad_lengths = [(int(math.ceil((i-1)/2)), int(math.floor((i-1)/2))) for i in self.kernel_size]

    if self.same_padding:
      array1 = np.pad(feature_map, pad_lengths)
    else:
      array1 = feature_map

    # kernel_gradient = self.convolve3d(array1, prev_grad)
    kernel_gradient = self.convolve_kernel(array1, prev_grad)
    self.kernels -= learning_rate * kernel_gradient

    # calculate new_prev_grad to backpropagate
    # flip over all axes (basically rotate by 180 degrees) 
    flipped_filter = np.flip(self.kernels)
    
    # have to pad in reverse
    if self.same_padding:
      # padded_prev_grad = np.pad(prev_grad, np.flip(pad_lengths))
      # don't want to flip the z pad length, should stay 0
      padded_prev_grad = np.pad(prev_grad, [pad_lengths[1], pad_lengths[0], pad_lengths[2]])
    else: # "full" convolution, pad by i-1 on both sides of gradient for i in kernel_size
      padded_prev_grad = np.pad(prev_grad, [(i-1, i-1) for i in self.kernel_size])
    
    new_prev_grad = np.zeros_like(feature_map)
    num_channels = self.kernel_size[2]
    for i in range(self.F): 
      new_prev_grad += self.convolve3d(padded_prev_grad[:, :, i*num_channels:(i+1)*num_channels], 
                                       flipped_filter[i])
    return new_prev_grad



from numpy import unravel_index

class PoolingLayer:
  # size is pool size
  # stride is amount that is skipped between starts of pools
  def __init__(self, size=(2,2), stride=(2,2)):
    self.size = size
    self.stride = stride

  # feature_map is nxnxF, F is number of filters
  def forward(self, feature_map):
    # print(feature_map.shape)
    x_size = int((feature_map.shape[0] - self.size[0])/self.stride[0]+1)
    y_size = int((feature_map.shape[1] - self.size[1])/self.stride[1]+1)
    return_map = np.empty((x_size, y_size, feature_map.shape[2]))
    
    cur_x = 0
    cur_y = 0
    for f in range(feature_map.shape[2]):
      for iter_x in range(x_size):
        for iter_y in range(y_size):
          cur_x = iter_x * self.stride[0]
          cur_y = iter_y * self.stride[1]
          pool = feature_map[cur_x:cur_x+self.size[0], cur_y:cur_y+self.size[1], f]
          # print(pool)
          return_map[iter_x, iter_y, f] = np.amax(pool)
          # print(np.amax(pool))
          # cur_x += self.stride[0]
          # cur_y += self.stride[1]
          # iter_x += 1
          # iter_y += 1
          # if cur_x >= feature_map.shape[0] or cur_y >= feature_map.shape[1]:
          #   break

    return return_map


  # prev grad is gradient for each pixel of the pooled layer, size of mxm
  # feature_map is original input into layer
  # want to return the backpropogated gradient, with only the max pixels having a change
  # every non-max pixel doesn't get changed, change of 0
  # basically "stretches out" prev_grad
  def backward(self, feature_map, prev_grad, learning_rate):
    x_size = prev_grad.shape[0]
    y_size = prev_grad.shape[1]

    cur_x = 0
    cur_y = 0

    new_prev_grad = np.zeros_like(feature_map)

    for f in range(feature_map.shape[2]):
      for iter_x in range(x_size):
        for iter_y in range(y_size):
          cur_x = iter_x * self.stride[0]
          cur_y = iter_y * self.stride[1]
          pool = feature_map[cur_x:cur_x+self.size[0], cur_y:cur_y+self.size[1], f]

          # get indices of maximum element in pool
          max_indices = unravel_index(pool.argmax(), pool.shape)

          new_prev_grad[max_indices[0] + cur_x, max_indices[1] + cur_y, f] = prev_grad[iter_x, iter_y, f]

    return new_prev_grad



# flattening layer for flattening the pooled nxnxf into 1x(n^2*f) layer

class FlatteningLayer:
  def __init__(self):
    return

  # feature map is nxnxf
  def forward(self, feature_map):
    tot_size = feature_map.size
    return_map = np.empty((1, tot_size))
    iter = 0
    for x in range(feature_map.shape[0]):
      for y in range(feature_map.shape[1]):
        for z in range(feature_map.shape[2]):
          return_map[0, iter] = feature_map[x, y, z]
          iter += 1
    return return_map

  # accept 1x(n^2*f) prev_grad and return nxnxf gradient
  def backward(self, feature_map, prev_grad, learning_rate):
    new_prev_grad = np.empty_like(feature_map)
    iter = 0
    for x in range(feature_map.shape[0]):
      for y in range(feature_map.shape[1]):
        for z in range(feature_map.shape[2]):
          new_prev_grad[x, y, z] = prev_grad[0, iter]
          iter += 1

    return new_prev_grad
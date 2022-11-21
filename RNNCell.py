from ActivationLayer import ActivationLayer 
import numpy as np

# A class for a cell of a Recurrent Neural Network
class RNNCell:

    # Class contains two weight matrices:
    # One, a hidden weight matrix, where W_h () will be multiplied
    # by the hidden state h_state (mx1)
    # Other, an input weight matrix, where W_x will be multiplied
    # by input x (nx1)

    # h_state_size is same as output
    # W_h @ h_state will come out to same dim as h_state, so
    # W_h should be mxm, and W_x should be mxn
    # g_1 is activation function to get next hidden state
    def __init__(self, h_state_size, input_size, g_1, g_1_deriv):
        self.h_state = np.zeros(shape=(h_state_size, 1))

        # matrix of weights to multiply by mx1 hidden state, so mxm size
        self.W_h = np.random.rand(h_state_size, h_state_size)
        
        # matrix of weights to multiply by nx1 input, so mxn size
        self.W_x = np.random.rand(h_state_size, input_size)
        
        # vector of bias values for new hidden state
        self.b_h = np.random.rand(h_state_size, 1)
        
        # make an activation layer from the given activation func
        # activation function for new hidden state
        self.activ_layer = ActivationLayer(activ_func=g_1, activ_func_deriv=g_1_deriv)
        
        
        # record all previous inputs, in sequential order
        self.prev_inputs = []


    # forward pass, get new hidden state and return hidden state
    def forward(self, x):

        # multiply input by weights and hidden state by hidden weights, add bias
        weighted_input = self.W_h @ self.h_state + self.W_x @ x +  self.b_h

        # perform activation function on weighted
        # to get new hidden state
        next_h_state = self.g_1(weighted_input)

        # update hidden state
        self.h_state = next_h_state
        return next_h_state

    # x is input to cell that resulted in prev_grad, nx1
    # prev_grad is mx1 vector of gradient of learning rate w.r.t. each
    # value of hidden state
    # single step of backwards
    def backward_step(self, x, prev_grad, learning_rate, t, vanishing_cutoff=0):
        
        # want to backpropagate through activation function, so
        # call the backward method for activation layer
        deactivated_grad = self.activ_layer.backward(x, prev_grad, learning_rate)
        
        hidden_weight_gradient = deactivated_grad @ np.transpose(self.h_state)
        input_weight_gradient = deactivated_grad @ np.transpose(x)
        bias_gradient = deactivated_grad

        # hidden state part: need to return gradient of loss
        # w.r.t current hidden state, to call backwards again
        hidden_state_gradient = np.transpose(self.W_h) @ deactivated_grad

        # bias gradient is same as this deactivated gradient
        self.W_h -= learning_rate * hidden_weight_gradient
        self.W_x -= learning_rate * input_weight_gradient
        self.b_h -= learning_rate * bias_gradient

        return hidden_state_gradient


    # input index is index of the current input to perform backward_step on
    # t is number of times to perform the backward step remaining
    # vanish is cutoff abs. value at which to stop backwards calls
    # if gradient ever goes below vanishing cutoff, stop
    # explode cutoff is abs. value at which to stop backwards calls
    def backward(self, input_index, prev_grad, learning_rate, t, vanish, explode):
        
        # if conditions not met, finish
        if input_index < 0 or t < 0:
            return
        
        # do vanish/explode condition

        new_grad = self.backward_step(self.prev_inputs[input_index], prev_grad, learning_rate)
        
        # do another backward step
        self.backward(input_index-1, new_grad, learning_rate, t-1, vanish, explode)

    # reset previous inputs
    def reset_input(self):
        self.prev_inputs = []

import numpy as np
from types import SimpleNamespace

class SimpleCell:
    """
    A simple rnn cell which will only be recursively connected to itself in the previous timestep.
    """

    # To be used for empirical gradient calculation
    H = 0.0005
    
    def __init__(self, ninputs, sigmoid=np.tanh, learning_rate=0.25):
        assert ninputs >= 1.0
        
        self.ninputs = ninputs
        self.sigmoid = sigmoid
        self.learning_rate = learning_rate

        weight_dist_range = np.sqrt(1. / ninputs)
        # Input weights
        self.U = np.random.uniform(-weight_dist_range, weight_dist_range, ninputs)
        # Output bias
        self.bias = np.random.uniform(-weight_dist_range, weight_dist_range)
        # Weight for recurrent connection
        self.w = np.random.uniform(-1., 1.)

        self.h = list([0])  # h[t] = h_t (h at time step t)
                            # Initialized with a zero so the first iteration has a t-1 to reference
    
    def forward_prop(self, x):
        time_steps = len(x)
        
        # Validate input size
        for xt in x:
            assert len(xt) == self.ninputs
        
        # Have an extra element so we dont have 't - 1' everywhere
        x = [None] + list(x)
        self.h = np.zeros(time_steps + 1)
        output = np.zeros(time_steps + 1)

        # Starts at t = 1, so h[t - 1] will be h[0] 
        for t in range(1, time_steps + 1):
            inpt = x[t]
            prev_h = self.h[t - 1]

            h = self.sigmoid(self.bias + np.dot(self.U, inpt) + self.w * prev_h)
            output[t] = h 
            self.h[t] = h

        return output[1:]

    def error(self, o, exp_o):
        """
        Calculates mean squared error
        """
        s = 0.
        for actual, expected in zip(o, exp_o):
            s += np.square(actual - expected)
        s /= len(o)
        return s

    def empirical_gradients(self, x, exp_o):
        bias = self.bias
        
        self.bias = bias + SimpleCell.H
        err_upper = self.error(self.forward_prop(x), exp_o)
        
        self.bias = bias - SimpleCell.H
        err_lower = self.error(self.forward_prop(x), exp_o)
        
        self.bias = bias
        d_bias = (err_upper - err_lower) / (2. * SimpleCell.H)

        d_U = np.zeros(self.ninputs)
        for weight_index in range(0, self.ninputs):
            weight = self.U[weight_index]
            
            self.U[weight_index] = weight + SimpleCell.H
            err_upper = self.error(self.forward_prop(x), exp_o)

            self.U[weight_index] = weight - SimpleCell.H
            err_lower = self.error(self.forward_prop(x), exp_o)

            d_weight = (err_upper - err_lower) / (2. * SimpleCell.H)
            d_U[weight_index] = d_weight
        
        return SimpleNamespace(d_bias=d_bias, d_U=d_U)

    def slow_learn(self, x, exp_o):
        gradients = self.empirical_gradients(x, exp_o)

        self.bias += -self.learning_rate * gradients.d_bias

        for weight_index in range(0, self.ninputs):
            self.U[weight_index] += -self.learning_rate * gradients.d_U[weight_index]

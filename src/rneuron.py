import numpy as np

class RNeuron:

    __ALL_NEURONS = []

    def __init__(self):
        RNeuron.__ALL_NEURONS.append(self)
        self.t = None
        self.input_neurons = []
        self.input = []
        self.output = 0.0
        self.sigma = np.tanh
        self.dsigma = lambda x: (1.0 - np.tanh(x)**2)
        self.input_weights = np.random.uniform(-1.0, 1.0, 0)

    def add_input(self, new_input, time_delay):
        self.input_neurons.append((new_input, time_delay))
        self.input_weights = np.append(self.input_weights, [np.random.uniform(-1.0, 1.0)])

    def propagate(self, t):
        print("AAA")
        if self.t == t:
            return
        print("BB")

        self.t = t

        input = np.zeros(len(self.input_neurons))
        self.input.append(input)
        
        # For all input neurons, propagate the input for this time step to them.
        for (neuron, time_delay), i in zip(self.input_neurons, range(len(input))):
            if neuron.t < t:
                # Ensure that the neuron is only 1 time step behind
                # I think it is okay to ignore the time delay here.
                assert neuron.t == t - 1
                neuron.propagate(t)
            
            # Might be able to use some trickery to get rid of the branch here
            if time_delay:
                if time_delay > t:
                    input[i] = 0.0
                else:
                    input[i] = neuron.hidden[t - time_delay]
            else:
                input[i] = neuron.output
        
        # input -> hidden_state[t]
        # store hidden_state[t]
        # hidden_state[t] -> output
        # store output

        self.output = self.sigma(sum(input))

        return self.output

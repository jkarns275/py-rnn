import numpy as np
from rneuron import RNeuron

class RNetwork:

    def __init__(self, ninputs, noutputs):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.input_neurons = [RNeuron(f"input_{i}") for i in range(ninputs)]
        self.output_neurons = [RNeuron(f"output_{i}") for i in range(noutputs)] 
        self.neurons = set(self.input_neurons + self.output_neurons)
    
    def neuron(self, name=None):
        n = RNeuron(name)
        self.neurons.add(n)
        return n

    def get_input_neuron(self, i):
        assert 0 <= i < self.ninputs
        return self.input_neurons[i]

    def get_output_neuron(self, i):
        assert 0 <= i < self.noutputs
        return self.output_neurons[i]

    def set_input(self, inputs, t):
        assert len(inputs) == self.ninputs
        
        for neuron, value in zip(self.input_neurons, inputs):
            neuron.t = t
            neuron.output = value
            neuron.outputs.append(value)
    
    def mse(self, exp_output, outputs):
        error = 0.0
        for exp, out in zip(exp_output, outputs):
            dif = np.subtract(exp, out)
            error += sum(np.square(dif))
        return error

    def back_propagate(self, inputs, exp_outputs, learning_rate=1.0):
        outputs = self.think(inputs)
        err = self.mse(exp_outputs, outputs)
        timesteps = len(inputs)

        gradients = {}
        for neuron in self.neurons:
            # At least 1 more than the final time step.
            # This is just book keeping, as neuron.t gets used to check whether or not neuron
            # has already calculated gradients for timestep t
            neuron.t = timesteps
            
            # Initialize gradients to zero
            gradients[neuron] = (np.zeros(neuron.w1.shape), np.zeros(neuron.w2.shape))
        
        for t in reversed(range(0, timesteps)):
            print(f"t = {t}")
            for i, neuron in enumerate(self.output_neurons):
                neuron.back_propagate(t, gradients, expected=exp_outputs[t][i])
        return gradients 
        # for neuron in self.neurons:
        #    g1, g2 = gradients[neuron]
        #    neuron.w1 -= g1 * learning_rate * err
        #    neuron.w2 -= g1 * learning_rate * err

    def empirical_learn(self, inputs, exp_outputs, err, learning_rate=0.425):
        gradients = {}
        h = 0.01
        h2 = 2 * h
        for neuron in self.neurons:
            w1_grads = np.zeros(neuron.w1.shape)
            for i in range(len(neuron.w1)):
                w0 = neuron.w1[i]
                neuron.w1[i] = w0 + h
                error_upper = self.mse(exp_outputs, self.think(inputs))
                neuron.w1[i] = w0 - h
                error_lower = self.mse(exp_outputs, self.think(inputs))
                w1_grads[i] = (error_upper - error_lower) / h2
            
            w2_grads = np.zeros(neuron.w2.shape)
            for i in range(len(neuron.w2)):
                w0 = neuron.w2[i]
                neuron.w2[i] = w0 + h
                error_upper = self.mse(exp_outputs, self.think(inputs))
                neuron.w2[i] = w0 - h
                error_lower = self.mse(exp_outputs, self.think(inputs))
                w2_grads[i] = (error_upper - error_lower) / h2
            
            gradients[neuron] = (w1_grads, w2_grads)
        return gradients 
        # for neuron, (w1_grads, w2_grads) in gradients.items():
        #     neuron.w1 -= w1_grads * learning_rate * err
        #     neuron.w2 -= w2_grads * learning_rate * err

    def think(self, inputs):
        assert len(inputs[0]) == self.ninputs
        for neuron in self.neurons:
            neuron.reset()
        outputs = []
        timesteps = len(inputs)
        for t in range(timesteps):
            self.set_input(inputs[t], t)
            for neuron in self.output_neurons:
                neuron.propagate(t)
            o = [on.output for on in self.output_neurons]
            outputs.append(o)
        return outputs

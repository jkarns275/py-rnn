import numpy as np
from rneuron import RNeuron

class RNetwork:

    def __init__(self, ninputs, noutputs):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.input_neurons = [RNeuron() for i in range(ninputs)]
        self.output_neurons = [RNeuron() for i in range(noutputs)] 

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

    def think(self, inputs):
        assert len(inputs[0]) == self.ninputs
        
        timesteps = len(inputs)
        for t in range(timesteps):
            self.set_input(inputs[t], t)
            for neuron in self.output_neurons:
                neuron.propagate(t)

        return [on.output for on in self.output_neurons]

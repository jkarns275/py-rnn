import numpy as np

class RNeuron:
    
    N_NEURONS = 0

    def __init__(self, name=None):
        self.t = -1
        self.name = name
        
        RNeuron.N_NEURONS += 1

        if name is None:
            name = f"neuron_{RNeuron.N_NEURONS}"

        # Even if there are no other recurrent connections, 
        # at least propagate the hidden state forward.
        self.rec_input_neurons = [(self, 0)] 
        self.input_neurons = []
        
        # Should contain tuples: (neuron, weight_index)
        self.output_neurons = []
        # Should contain tuples: (neuron, time_delay, weight_index)
        self.rec_output_neurons = [(self, 0, 0)]

        self.hidden_net = []
        self.hidden = []
        self.net = []
        self.input = []
        self.output = 0.0
        self.outputs = []
        self.sigma = np.tanh
        self.dsigma = lambda x: (1.0 - np.tanh(x)**2)
        self.deltas = np.zeros(len(self.outputs))
        self.hidden_deltas = np.zeros(len(self.outputs))

       
        # Weights for normal inputs. First one is bias
        self.w1 = np.random.uniform(-1.0, 1.0, 1)
        # Weights for recurrent connections.
        # The first is for bias, 
        # the second initial weight is for the connection to self with time delay 0,
        # which is just used to propagate the hidden state
        self.w2 = np.random.uniform(-1.0, 1.0, 2)

    def reset(self):
        self.t = -1
        self.hidden_net = []
        self.hidden = []
        self.net = []
        self.input = []
        self.output = 0.0
        self.outputs = []
        self.deltas = None
        self.hidden_deltas = None

    def __add_rec_output(self, new_output, time_delay, weight_index):
        self.rec_output_neurons.append((new_output, time_delay, weight_index))
    
    def __add_output(self, new_output, weight_index):
        self.output_neurons.append((new_output, weight_index))

    def add_input(self, new_input, time_delay):
        if time_delay != 0:
            weight_index = len(self.w2)
            self.rec_input_neurons.append((new_input, time_delay))
            self.w2 = np.append(self.w2, [np.random.uniform(-1.0, 1.0)])
            new_input.__add_rec_output(self, time_delay, weight_index)
        else:
            weight_index = len(self.w1)
            self.input_neurons.append(new_input)
            self.w1 = np.append(self.w1, [np.random.uniform(-1.0, 1.0)])
            new_input.__add_output(self, weight_index)

    def back_propagate(self, t, gradients, expected=None):
        # Ignore input nodes
        if len(self.hidden) == 0:
            return
        if self.t <= t:
            # Already been been visited
            return self.deltas[t]
        if self.deltas is None:
            self.deltas = np.zeros(len(self.outputs))
            self.hidden_deltas = np.zeros(len(self.outputs))

        self.t = t
        
        # Make sure deltas have been calculated for all output neurons since they are needed for the
        # backprop calculation.
        for neuron, _ in self.output_neurons:
            neuron.back_propagate(t, gradients)

        for neuron, _, _ in self.rec_output_neurons:
            neuron.back_propagate(t, gradients)

        # Expected output will only be set for output neurons
        # Only deltas for the hidden connections will be calculated as output neurons
        # because they're the only ones we know the proper output for.
        if expected is not None:
            assert expected is not None
            
            # derivative of error with respect to the output
            dErr = self.outputs[t] - expected

            # derivative of the output with respect to hidden_net
            dOut = self.dsigma(self.hidden_net[t])

            self.hidden_deltas[t] = dErr * dOut
        else:
            # Sum of error propagated from this neuron
            dErr = 0.0
            for (neuron, weight_index) in self.output_neurons:
                dErr += neuron.deltas[t] * neuron.w1[weight_index]

            # derivative of the output with respect to hidden_net
            dOut = self.dsigma(self.hidden_net[t])
            self.hidden_deltas[t] = dErr * dOut

        # Now calculate the first delta
        # Since this delta is "inside" the cell it is never a direct output and
        # will always use the generic back prop calculation
        dErr = 0.0
        for (neuron, time_delay, weight_index) in self.rec_output_neurons:
            if t + time_delay < len(neuron.hidden):
                dErr += neuron.hidden_deltas[t + time_delay] * neuron.w2[weight_index]
        
        dOut = self.dsigma(self.net[t])

        self.deltas[t] = dErr * dOut

        g1, g2 = gradients[self]
        for i in range(len(g1) - 1):
            g1[i + 1] += self.input_neurons[i].outputs[t] * self.deltas[t]
        
        g1[0] = self.deltas[t]

        for i in range(len(g2) - 1):
            neuron, time_delay = self.rec_input_neurons[i]
            if time_delay + t < len(neuron.hidden_deltas):
                g2[i + 1] += neuron.outputs[t] * self.hidden_deltas[t]
        g2[0] = self.hidden_deltas[t]
        # Since this neuron is done with backprop, continue propagating backwards!
        for neuron in self.input_neurons:
            neuron.back_propagate(t, gradients)
        for neuron, _ in self.rec_input_neurons:
            neuron.back_propagate(t, gradients)
    
    def propagate(self, t):
        if self.t == t:
            return

        self.t = t

        input = np.zeros(len(self.input_neurons))
        self.input.append(input)
                
        # For all input neurons, propagate the input for this time step to them.
        for neuron, i in zip(self.input_neurons, range(len(input))):
            if neuron.t < t:
                # Ensure that the neuron is only 1 time step behind
                # I think it is okay to ignore the time delay here.
                assert neuron.t == t - 1
                neuron.propagate(t)
            
            input[i] = neuron.output
        
        # print(f"{self.name}.input = {input}")        
        net = self.w1[0] + np.dot(self.w1[1:], input)
        self.net.append(net)
        # bias + weights * input
        self.hidden.append(self.sigma(net))
        
        hidden_input = np.zeros(len(self.rec_input_neurons))
        hidden_input[0] = self.hidden[-1]
        for (neuron, time_delay), i in zip(self.rec_input_neurons, range(len(hidden_input))):
            if neuron.t < t:
                # Same reasoning as above
                assert neuron.t == t - 1
                neuron.propagate(t)
                
            if t >= time_delay:
                hidden_input[i] = neuron.hidden[-time_delay - 1]
 
        # print(f"{self.name}.hidden_input = {hidden_input}")
        hidden_net = self.w2[0] + np.dot(self.w2[1:], hidden_input)
        self.hidden_net.append(hidden_net)
        self.output = self.sigma(hidden_net)
        self.outputs.append(self.output)

        # print(f"{self.name}.output = {self.output}")        
        return self.output

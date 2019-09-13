import numpy as np
import matplotlib.pyplot as plt
from rnn import RNetwork
from rneuron import RNeuron

def make_test_net(n):
    rnn = RNetwork(1, 1)
    n1 = rnn.neuron("kyle")
    n2 = rnn.neuron("nick")
    n2.add_input(rnn.get_input_neuron(0), 0)
    n1.add_input(n2, 0)
    n1.add_input(n2, 1)
    rnn.get_output_neuron(0).add_input(n1, 0)
    return rnn

def fully_connected(n):
    rnn = RNetwork(4, 1)
    neurons = [rnn.neuron(f"{i}") for i in range(n)]
    for neuron in neurons:
        for n in neurons:
            neuron.add_input(n, 1)
        for i in range(4):
            neuron.add_input(rnn.get_input_neuron(i), 0)
        rnn.get_output_neuron(0).add_input(neuron, 0)

    return rnn

def main():
    target = fully_connected(7)
    h = fully_connected(7)
    err = 1
    while err > .01: 
        inputs = [np.random.uniform(-1, 1, 4) for _ in range(1)]
        exp = target.think(inputs)
        err = h.mse(exp, h.think(inputs))
        print(f"err = {err}")
        brads = h.back_propagate(inputs, exp)
    
if __name__ == "__main__":
    main()

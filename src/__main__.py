import numpy as np
import matplotlib.pyplot as plt
from rnn import RNetwork

def main():
    rnn = RNetwork(1, 1)
    rnn.get_output_neuron(0).add_input(rnn.get_input_neuron(0), 0)
    print(rnn.think([[1.0]]))


if __name__ == "__main__":
    main()

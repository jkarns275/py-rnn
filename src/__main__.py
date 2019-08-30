from simple_cell import SimpleCell
import numpy as np
import matplotlib.pyplot as plt

def main():
    mycell = SimpleCell(1, learning_rate=0.125, n_nodes=2)
    
    training_input = [[i / 100 * (2 * np.pi)] for i in range(100)]
    training_outpt = [(np.cos(i) + 1) / 2 for i in training_input]

    x = [i[0] for i in training_input]

    for i in range(0, 200):
        mycell.slow_learn(training_input, training_outpt)
        output = mycell.forward_prop(training_input)
        error = mycell.error(output, training_outpt)
        print(f"error = {error[0]}")
    plt.plot(x, output)
    plt.plot(x, training_outpt)
    plt.show()
if __name__ == "__main__":
    main()

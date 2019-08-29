from simple_cell import SimpleCell
import numpy as np
import matplotlib.pyplot as plt

def main():
    mycell = SimpleCell(1, learning_rate=0.25)

    training_input = [[i / 70 * (4 * np.pi)] for i in range(70)]
    training_outpt = [np.sin(i) for i in training_input]

    for i in range(0, 500):
        mycell.slow_learn(training_input, training_outpt)
        output = mycell.forward_prop(training_input)
        error = mycell.error(output, training_outpt)
        print(f"error = {error[0]}")
    x = [i[0] for i in training_input]
    y = [np.sin(i) for i in x]
    plt.plot(x, output)
    plt.plot(x, y)
    plt.show()
if __name__ == "__main__":
    main()

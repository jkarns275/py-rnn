from simple_cell import SimpleCell
import numpy as np

def main():
    mycell = SimpleCell(1, learning_rate=0.125/4)

    training_input = [[i / 20 * (2 * np.pi)] for i in range(20)]
    training_outpt = [np.sin(i) for i in training_input]

    for i in range(0, 500000):
        mycell.slow_learn(training_input, training_outpt)
        output = mycell.forward_prop(training_input)
        error = mycell.error(output, training_outpt)
        print(f"error = {error[0]}\noutput = {output}")
            
if __name__ == "__main__":
    main()

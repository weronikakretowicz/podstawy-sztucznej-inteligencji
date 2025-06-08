import csv

import numpy as np

FILENAME = "weights.csv"


class NeuralNetwork:
    def __init__(self, num_of_in_nodes):
        self.nodes_in_layers = []
        self.nodes_in_layers.append(num_of_in_nodes)

    def add_weights(self, filename, matrix_of_weights):
        with open(filename, 'a', newline='') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerows(matrix_of_weights)

            f_object.close()

    def load_weights(self, filename):
        matrix = []

        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                matrix.append(np.array(row, dtype=float))

        return matrix

    def add_layer(self, num_of_nodes, min_value=-1, max_value=1):
        matrix_of_weights = []

        for i in range(0, num_of_nodes):
            row_weights = np.random.uniform(low=min_value, high=max_value, size=self.nodes_in_layers[-1])
            matrix_of_weights.append(row_weights)

        self.nodes_in_layers.append(num_of_nodes)
        self.add_weights(FILENAME, matrix_of_weights)

    def check_if_properties_are_correct(self, matrix, vector):
        columns = len(matrix[0])
        rows = len(vector)

        if rows != columns:
            return 1
        else:
            return 0

    def calculate_layer_output(self, matrix, vector):
        if self.check_if_properties_are_correct(matrix, vector):
            return 1
        else:
            return np.dot(matrix, vector)

    def predict(self, input_vector):
        curr_vector = input_vector
        num_of_layers = len(self.nodes_in_layers) - 1
        matrix_of_weights = self.load_weights(FILENAME)
        curr_matrix = []
        last_row = 0
        index_row = 0

        for i in range(0, num_of_layers):
            while (index_row < (last_row + self.nodes_in_layers[i + 1])):
                curr_matrix.append(matrix_of_weights[index_row])
                index_row += 1

            last_row = index_row
            curr_vector = self.calculate_layer_output(curr_matrix, curr_vector)
            if type(curr_vector) == int:
                return 1

            curr_vector = curr_vector.T
            curr_matrix.clear()

        return curr_vector


def main():
    features = np.array([0.5, 0.75, 0.1]).T
    network = NeuralNetwork(3)

    np.random.seed(1)
    network.add_layer(4)
    network.add_layer(2)
    network.add_layer(7)

    output = network.predict(features)
    if (type(output) == int):
        print(f"Incorrect dimensions!\n")
    else:
        print(f"Output vector: {output}\n")


if __name__ == "__main__":
    main()

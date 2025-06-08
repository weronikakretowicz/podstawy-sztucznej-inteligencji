import csv

import numpy as np

LEARNING_RATE = 0.01
FILE_NAME = 'weights_zad4.csv'


class NeuralNetworkColors:
    def __init__(self):
        self.nodes_in_layers = [3, 6, 4]
        self.generate_weights()

    def generate_weights(self):
        for i in range(1, len(self.nodes_in_layers)):
            matrix_of_weights = []

            for j in range(0, self.nodes_in_layers[i]):
                row_weights = np.random.uniform(low=-1, high=1, size=self.nodes_in_layers[i - 1])
                matrix_of_weights.append(row_weights)

            self.save_weights_to_file(matrix_of_weights)

    def save_weights_to_file(self, matrix_of_weights):
        with open(FILE_NAME, 'a', newline='') as f_object:
            writer_object = csv.writer(f_object, delimiter=' ')
            writer_object.writerows(matrix_of_weights)
            writer_object.writerow([])

            f_object.close()

    def activate_relu(self, layer_output):
        for val in range(0, len(layer_output)):
            layer_output[val] = max(0, layer_output[val])

        return layer_output

    def relu_derivative(self, layer_delta):
        data = [1 if value > 0 else 0 for value in layer_delta]

        return np.array(data)

    def predict(self, input_vector, weights):
        curr_vector = input_vector
        num_of_layers = len(self.nodes_in_layers) - 1

        for i in range(0, num_of_layers):
            curr_vector = self.calculate_layer_output(weights[i], curr_vector, i)
            curr_vector = curr_vector.reshape(-1, 1)

        return curr_vector

    def calculate_layer_output(self, matrix, vector, layer_index):
        res = np.dot(matrix, vector)

        if layer_index == 1:
            res = self.activate_relu(res)

        return np.array(res)

    def count_error(self, my_output, proper_output):
        return np.subtract(my_output, proper_output) ** 2

    def calculate_h_layer_output(self, features, weights):
        output_vector = np.dot(weights[0], features)

        return output_vector.reshape(-1, 1)

    def calculate_o_delta(self, res, expected_result):
        nodes_in_o_layer = len(expected_result)
        res = (2 / nodes_in_o_layer * np.subtract(res, expected_result))

        return np.array(res)

    def calculate_h_delta(self, output_layer_delta, weights):
        weights = np.array(weights)
        weights = weights.T

        res = np.dot(weights, output_layer_delta)
        return np.array(res)

    def multiply_vectors(self, h_layer_delta, vector):
        for i in range(0, len(vector)):
            h_layer_delta[i] = np.abs(h_layer_delta[i] * vector[i])

        return h_layer_delta

    def calculate_delta_after_derivative(self, h_layer_delta, h_layer_output):
        val = self.relu_derivative(h_layer_output)

        return self.multiply_vectors(h_layer_delta, val)

    def rescale_delta(self, layer_delta, prev_layer_output):
        res = np.multiply(layer_delta, prev_layer_output)
        return res

    def set_new_weights(self, all_weights, proper_layers_deltas):
        num_of_layers = len(all_weights)
        new_weights = []

        for i in range(0, num_of_layers):
            weights = np.subtract(all_weights[i], LEARNING_RATE * proper_layers_deltas[i])
            new_weights.append(weights)

        return new_weights

    def count_deltas(self, output_res, expected_output_res, all_weights, input_data):
        proper_layers_deltas = []
        calculated_deltas = []

        last_h_layer_index = len(self.nodes_in_layers) - 2
        o_layer_delta = self.calculate_o_delta(output_res, expected_output_res).reshape(-1, 1)
        calculated_deltas.append(o_layer_delta)
        last_delta = o_layer_delta

        while last_h_layer_index > 0:
            h_layer_delta = (self.calculate_h_delta(last_delta, all_weights[last_h_layer_index])).reshape(-1, 1)
            # self.calculate_h_layer_output(input_data, all_weights) <- zwraca kolumnę
            h_layer_delta = np.array(self.calculate_delta_after_derivative(h_layer_delta,
                                                                           self.calculate_h_layer_output(input_data,
                                                                                                         all_weights)))

            calculated_deltas.append(h_layer_delta)
            last_delta = h_layer_delta
            last_h_layer_index -= 1

        o_layer_delta = calculated_deltas[0]
        proper_o_delta = np.array(
            self.rescale_delta(o_layer_delta, (self.calculate_h_layer_output(input_data, all_weights)).reshape(1, -1)))
        proper_layers_deltas.append(proper_o_delta)

        h_layer_delta = calculated_deltas[1]
        proper_h_delta = np.array(self.rescale_delta(h_layer_delta, input_data.reshape(1, -1)))
        proper_layers_deltas.append(proper_h_delta)

        # kolejność od pierwszej wartwy ukrytej
        proper_layers_deltas.reverse()

        return proper_layers_deltas

    def load_weights(self):
        matrix_of_weights = []

        with open(FILE_NAME, "r") as csvfile:
            for layer in range(1, len(self.nodes_in_layers)):

                curr_matrix = []
                for node in range(self.nodes_in_layers[layer]):
                    line = csvfile.readline()
                    clean_line = line.rstrip('\n')

                    float_numbers = []
                    for num in clean_line.split(' '):
                        float_numbers.append(float(num))

                    curr_row = np.array(float_numbers)
                    curr_matrix.append(curr_row)

                matrix_of_weights.append(curr_matrix)
                csvfile.readline()

            csvfile.close()

        return matrix_of_weights

    def load_data(self, train_file):
        all_features = []
        all_proper_outputs = []
        output_mapping = {
            '1': [1, 0, 0, 0],
            '2': [0, 1, 0, 0],
            '3': [0, 0, 1, 0],
            '4': [0, 0, 0, 1]
        }

        with open(train_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')

            for row in csv_reader:
                features = np.array([float(row[0]), float(row[1]), float(row[2])])

                if row[3] in output_mapping:
                    proper_output = np.array(output_mapping[row[3]])

                all_features.append(features)
                all_proper_outputs.append(proper_output)

            csv_file.close()

        return all_features, all_proper_outputs

    def save_updated_weights(self, weights):
        with open(FILE_NAME, 'w', newline='') as f_object:
            writer_object = csv.writer(f_object, delimiter=' ')

            for i in range(0, len(weights)):
                writer_object.writerows(weights[i])
                writer_object.writerow([])

            f_object.close()

    def train_neural_network(self, train_file):
        features, proper_output = self.load_data(train_file)
        weights = self.load_weights()
        num_of_series = len(features)
        print(len(features))

        for i in range(0, 50):
            for index in range(0, num_of_series):
                # zwraca kolumnę
                my_output = self.predict((features[index]).reshape(-1, 1), weights)
                error = sum(self.count_error(my_output, proper_output[index].reshape(-1, 1)))
                print(error)

                proper_deltas = self.count_deltas(my_output, proper_output[index].reshape(-1, 1), weights,
                                                  (features[index]).reshape(-1, 1))
                weights = self.set_new_weights(weights, proper_deltas)

            self.save_updated_weights(weights)

        return weights

    def find_equal_index(self, vector, value):
        for i in range(0, len(vector)):
            if vector[i] == value:
                return i

    def is_output_correct(self, my_output, proper_output):
        max_value = max(my_output)
        my_output_index = self.find_equal_index(my_output, max_value)
        proper_output_index = self.find_equal_index(proper_output, 1)

        return 1 if my_output_index == proper_output_index else 0

    def test_neural_network(self, test_file):
        features, proper_output = self.load_data(test_file)
        weights = self.load_weights()
        num_of_series = len(features)
        num_of_good_outputs = 0

        for i in range(0, num_of_series):
            my_output = self.predict(features[i].reshape(-1, 1), weights)
            if self.is_output_correct(my_output, proper_output[i].reshape(-1, 1)) == 1:
                num_of_good_outputs += 1

        return num_of_good_outputs, num_of_series


def main():
    np.random.seed(1)
    colors = NeuralNetworkColors()

    train_file = 'train.csv'
    colors.train_neural_network(train_file)

    test_file = 'test.csv'
    num_of_good_outputs, num_of_series = colors.test_neural_network(test_file)
    print(f"Number of good answers: {num_of_good_outputs} from {num_of_series} inputs")


if __name__ == "__main__":
    main()

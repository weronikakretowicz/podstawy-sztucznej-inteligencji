import csv
import struct

import numpy as np

FILENAME_WEIGHTS = "weights.csv"

LEARNING_RATE = 0.01
BATCH_SIZE = 100
NUM_OF_SERIES = 100
NODES_IN_H_LAYER = 40
NODES_IN_O_LAYER = 10


def read_idx_file(file_name):
    with open(file_name, 'rb') as f:
        # f.read(8) <- 8 bajtów, '>' big endian
        magic_number, num_items = struct.unpack('>II', f.read(8))

        if magic_number == 2051:  # Obrazy
            num_rows, num_cols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  # Etykiety
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, 1)

    return data


class NeuralNetwork:
    def __init__(self, num_of_in_nodes):
        self.nodes_in_layers = []
        self.activation_functions = {}
        self.nodes_in_layers.append(num_of_in_nodes)

    def add_network_structure_to_file(self, filename, matrix_of_weights, num_of_nodes, layer_index):
        with open(filename, 'a', newline='') as f_object:
            writer_object = csv.writer(f_object, delimiter=' ')

            activation_function = "None"
            if layer_index in self.activation_functions:
                activation_function = self.activation_functions[layer_index]

            writer_object.writerow([num_of_nodes])
            writer_object.writerow([activation_function])
            writer_object.writerows(matrix_of_weights)

            f_object.close()

    def save_to_file_new_data(self, filename, weights):
        with open(filename, 'w', newline='') as f_object:
            writer_object = csv.writer(f_object, delimiter=' ')

            for i in range(1, len(self.nodes_in_layers)):
                writer_object.writerow([self.nodes_in_layers[i]])

                if i in self.activation_functions:
                    writer_object.writerow([self.activation_functions[i]])
                else:
                    writer_object.writerow(["None"])

                writer_object.writerows(weights[i - 1])

            f_object.close()

    def load_weights(self, filename):
        matrix_of_weights = []

        with open(filename, "r") as csvfile:
            for layer in range(1, len(self.nodes_in_layers)):
                for i in range(0, 2):
                    csvfile.readline()

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

            csvfile.close()

        return matrix_of_weights

    def add_layer(self, num_of_nodes, activation_function=None, derivative_of_activate_fun=None, min_value=-0.1,
                  max_value=0.1):
        matrix_of_weights = []

        for i in range(0, num_of_nodes):
            row_weights = np.random.uniform(low=min_value, high=max_value, size=self.nodes_in_layers[-1])
            matrix_of_weights.append(row_weights)

        self.nodes_in_layers.append(num_of_nodes)

        layer_index = len(self.nodes_in_layers) - 1
        if activation_function is not None:
            self.activation_functions[layer_index] = activation_function, derivative_of_activate_fun

        self.add_network_structure_to_file(FILENAME_WEIGHTS, matrix_of_weights, num_of_nodes, layer_index)

    def dropout(self, output_vector, dropout_mask):
        output_vector = np.array(output_vector).reshape(1, -1)
        output_vector = np.array(np.multiply(output_vector, dropout_mask))

        return output_vector

    def calculate_layer_output(self, weights, input_vector, layer_index, dropout_mask):
        res = np.dot(weights, input_vector)

        if layer_index in self.activation_functions:
            my_list = self.activation_functions[layer_index]
            activate = my_list[0]
            res = activate(res)

        if layer_index == 1:
            res = self.dropout(res, dropout_mask) * 2

        return np.array(res)

    def activate_relu(self, vector):
        for val in range(0, len(vector)):
            vector[val] = max(0, vector[val])

        return vector

    def relu_derivative(self, vector):
        data = [1 if value > 0 else 0 for value in vector]

        return np.array(data)

    def predict(self, input_vector, weights, dropout_mask):
        curr_vector = input_vector
        num_of_layers = len(self.nodes_in_layers)

        for i in range(1, num_of_layers):
            curr_vector = self.calculate_layer_output(weights[i - 1], curr_vector, i, dropout_mask)
            curr_vector = curr_vector.reshape(-1, 1)

        return curr_vector

    def calculate_h_layer_output(self, features, weights):
        output_vector = np.dot(weights[0], features)

        return output_vector

    def calculate_o_delta(self, res, expected_result):
        nodes_in_o_layer = len(expected_result)
        res = (2 / nodes_in_o_layer * np.subtract(res, expected_result))

        return np.array(res)

    def calculate_h_delta(self, output_layer_delta, weights):
        output_layer_delta = output_layer_delta.reshape(-1, 1)
        weights = np.array(weights)
        weights = weights.T

        res = np.dot(weights, output_layer_delta)
        return np.array(res)

    def calculate_delta_after_derivative(self, h_layer_delta, h_layer_output):
        val = np.array(self.relu_derivative(h_layer_output)).reshape(-1, 1)
        res = np.multiply(h_layer_delta, val)

        return res

    def rescale_delta(self, layer_delta, prev_layer_output):
        layer_delta = layer_delta.reshape(-1, 1)
        res = np.multiply(layer_delta, prev_layer_output)

        return res

    def set_new_weights(self, all_weights, proper_layers_deltas):
        num_of_layers = len(all_weights)
        new_weights = []

        for i in range(0, num_of_layers):
            weights = all_weights[i] - LEARNING_RATE * proper_layers_deltas[i]
            new_weights.append(weights)

        return new_weights

    def count_deltas(self, output_res, expected_output_res, all_weights, all_outputs, vector):
        proper_layers_deltas = []
        calculated_deltas = []

        last_h_layer_index = len(self.nodes_in_layers) - 2
        o_layer_delta = self.calculate_o_delta(output_res, expected_output_res).reshape(-1, 1)
        calculated_deltas.append(o_layer_delta)
        last_delta = o_layer_delta

        while last_h_layer_index > 0:
            h_layer_delta = (self.calculate_h_delta(last_delta, all_weights[last_h_layer_index])).reshape(-1, 1)

            if last_h_layer_index in self.activation_functions:
                h_layer_delta = np.array(
                    self.calculate_delta_after_derivative(h_layer_delta, all_outputs[last_h_layer_index]))
            h_layer_delta = self.dropout(h_layer_delta, vector)

            calculated_deltas.append(h_layer_delta)
            last_delta = h_layer_delta
            last_h_layer_index -= 1

        input_index = len(calculated_deltas) - 1
        for i in range(0, len(calculated_deltas)):
            layer_delta = calculated_deltas[i]
            proper_delta = np.array(self.rescale_delta(layer_delta, all_outputs[input_index].reshape(1, -1)))
            # print(np.isnan(proper_delta).any(), epoc, seria)
            proper_layers_deltas.append(proper_delta)

            input_index -= 1

        # # kolejność od pierwszej wartwy ukrytej
        proper_layers_deltas.reverse()

        return proper_layers_deltas

    def count_error(self, my_output, proper_output):
        return np.subtract(my_output, proper_output) ** 2

    def fit(self, expected_results, features):
        num_of_layers = len(self.nodes_in_layers) - 1
        weights = self.load_weights(FILENAME_WEIGHTS)

        for epoc in range(0, 1000):
            # num_of_series = len(features[0])
            sum_of_errors = 0
            num_of_good_outputs = 0

            for i in range(0, NUM_OF_SERIES):
                dropout_mask = np.random.binomial(1, 0.5, NODES_IN_H_LAYER)
                all_outputs = []
                curr_input = np.array(features[:, i]).reshape(-1, 1)
                expected_output = np.array(expected_results[:, i]).reshape(-1, 1)
                all_outputs.append(curr_input)

                total_output = np.array(self.predict(curr_input, weights, dropout_mask)).reshape(-1, 1)
                output_max = self.find_index_with_max_value(total_output)
                expected_output_max = self.find_index_with_max_value(expected_output)

                if expected_output_max == output_max:
                    num_of_good_outputs += 1

                error = sum(self.count_error(total_output, expected_output))
                sum_of_errors += error

                for j in range(0, num_of_layers):
                    curr_input = (self.calculate_layer_output(weights[j], curr_input, j + 1, dropout_mask)).reshape(-1,
                                                                                                                    1)

                    all_outputs.append(curr_input)

                proper_layers_deltas = self.count_deltas(curr_input, expected_output,
                                                         weights, all_outputs, dropout_mask)
                weights = self.set_new_weights(weights, proper_layers_deltas)

            if epoc % 100 == 0:
                accuracy = (num_of_good_outputs / NUM_OF_SERIES * 100)
                print(f"Accuracy for epoc {epoc}: {accuracy}%\n")

            self.save_to_file_new_data(FILENAME_WEIGHTS, weights)

    def load_mnist_data(self):
        train_images = read_idx_file('MNIST_ORG/train-images.bin')
        train_labels = read_idx_file('MNIST_ORG/train-labels.bin')
        test_images = read_idx_file('MNIST_ORG/t10k-images.bin')
        test_labels = read_idx_file('MNIST_ORG/t10k-labels.bin')

        train_images = train_images / 255
        test_images = test_images / 255

        return train_images, train_labels, test_images, test_labels

    def find_index_with_max_value(self, my_output):
        max_val_index = 0

        for j in range(0, len(my_output)):
            if my_output[j] > my_output[max_val_index]:
                max_val_index = j

        return max_val_index

    def test_neural_network(self, features, expected_output, filename):
        num_of_good_outputs = 0
        weights = self.load_weights(filename)

        num_of_series = 1000  # len(features[0])
        for i in range(0, num_of_series):
            image = np.array(features[:, i]).reshape(-1, 1)
            curr_vector = image
            num_of_layers = len(self.nodes_in_layers)

            for j in range(1, num_of_layers):
                res = np.dot(weights[j - 1], curr_vector)

                if j in self.activation_functions:
                    my_list = self.activation_functions[j]
                    activate = my_list[0]
                    res = activate(res)

                curr_vector = np.array(res).reshape(-1, 1)

            output_max = self.find_index_with_max_value(curr_vector)
            expected_output_max = self.find_index_with_max_value(expected_output[:, i])

            if expected_output_max == output_max:
                num_of_good_outputs += 1

        return num_of_good_outputs


def main():
    # features = np.array([
    #     [0.5, 0.1, 0.2, 0.8],
    #     [0.75, 0.3, 0.1, 0.9],
    #     [0.1, 0.7, 0.6, 0.2]
    # ])
    #
    # weights_h = np.array([
    #     [0.1, 0.1, -0.3],
    #     [0.1, 0.2, 0.0],
    #     [0.0, 0.7, 0.1],
    #     [0.2, 0.4, 0.0],
    #     [-0.3, 0.5, 0.1]
    # ])
    #
    # weights_y = np.array([
    #     [0.7, 0.9, -0.4, 0.8, 0.1],
    #     [0.8, 0.5, 0.3, 0.1, 0.0],
    #     [-0.3, 0.9, 0.3, 0.1, -0.2],
    # ])
    #
    # weights = [weights_h, weights_y]
    #
    # proper_output = np.array([[
    #     0.1, 0.5, 0.1, 0.7],
    #     [1.0, 0.2, 0.3, 0.6],
    #     [0.1, -0.5, 0.2, 0.2],
    # ])
    #
    network = NeuralNetwork(784)
    np.random.seed(1)
    network.add_layer(NODES_IN_H_LAYER, activation_function=network.activate_relu)
    network.add_layer(NODES_IN_O_LAYER)

    train_images, train_labels, test_images, test_labels = network.load_mnist_data()

    all_images_train = []
    for sublist in train_images:
        flat_list = []

        for item in sublist:
            flat_list.append(item)
        flat_list = np.concatenate(flat_list)
        flat_list = flat_list.reshape(-1)
        all_images_train.append(flat_list)

    features = (np.array(all_images_train)).T

    # features = train_images.reshape(-1, len(train_images))
    expected_res = np.zeros((10, len(train_labels)))  # 10 klas w MNIST
    for i, label in enumerate(train_labels):
        expected_res[label, i] = 1

    # features_test = test_images.reshape(-1, len(test_images))
    expected_res_test = np.zeros((10, len(test_labels)))  # 10 klas w MNIST

    for i, label in enumerate(test_labels):
        expected_res_test[label, i] = 1

    all_images_test = []
    for sublist in test_images:
        flat_list = []

        for item in sublist:
            flat_list.append(item)
        flat_list = np.concatenate(flat_list)
        flat_list = flat_list.reshape(-1)
        all_images_test.append(flat_list)

    features_test = (np.array(all_images_test)).T

    network.fit(expected_res, features)

    num_of_good_outputs = network.test_neural_network(features_test, expected_res_test, FILENAME_WEIGHTS)
    print(f"Accuracy of good outputs in test: {num_of_good_outputs / 1000 * 100}%\n")

if __name__ == "__main__":
    main()

import csv
import struct
import math

import numpy as np

FILENAME_WEIGHTS = "weights.csv"

LEARNING_RATE = 0.2
BATCH_SIZE = 100
NUM_OF_SERIES = 1000
NODES_IN_H_LAYER = 100
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
        self.clear_file()
        self.nodes_in_layers = []
        self.activation_functions = {}
        self.nodes_in_layers.append(num_of_in_nodes)

    def clear_file(self):
        f = open(FILENAME_WEIGHTS, "w+")
        f.close()

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

    def dropout(self, outputs, dropout_masks):
        dropout_masks = np.array(dropout_masks).T
        output_vector = np.multiply(outputs, dropout_masks)

        return output_vector

    def calculate_layer_output(self, weights, inputs, layer_index, dropout_masks):
        # weights = (np.array(weights)).T
        # inputs = inputs.T
        res = np.dot(weights, inputs)
        # weights = (np.array(weights))
        # first = np.dot(weights, inputs[:, 0])
        # second = np.dot(weights, inputs[:, 1])

        after_activation = []
        if layer_index in self.activation_functions:
            my_list = self.activation_functions[layer_index]
            activate = my_list[0]

            for i in range(len(res[0])):
                activated = activate(res[:, i])
                activated = activated.reshape(-1)
                # if layer_index == 2:
                #     print(sum(activated))
                after_activation.append(activated)

            res = np.array(after_activation)
            res = res.T

        after_dropout = []
        if layer_index == 1:
            for i in range(len(res[0])):
                drop = self.dropout(after_activation[i], dropout_masks[i]) * 2
                drop = drop.reshape(-1)
                after_dropout.append(drop)

            final = np.array(after_dropout)
            final = final.T
            return final

        return np.array(res)

    def activate_relu(self, vector):
        for val in range(0, len(vector)):
            vector[val] = max(0, vector[val])

        return vector

    def relu_derivative(self, layer_outputs):
        after_derivative = []

        for vector in layer_outputs:
            row = [1 if value > 0 else 0 for value in vector]
            after_derivative.append(row)

        return np.array(after_derivative)

    def active_sigmoid(self, vector):
        minus_vector = (-1) * vector
        ones = (np.ones(len(vector))).reshape(-1, 1)

        exp_minus_x = []
        for i in range(len(minus_vector)):
            exp_minus_x.append(math.exp(minus_vector[i]))
        exp_minus_x = (np.array(exp_minus_x)).reshape(-1, 1)

        res = ones / (ones + exp_minus_x)

        return res

    def sigmoid_derivative(self, vector):
        ones = (np.ones(len(vector))).reshape(-1, 1)
        res = vector * (ones - vector)

        return res

    def active_hyperbolic_tangent(self, vector):
        minus_vector = (-1) * vector

        exp_x = []
        for i in range(len(vector)):
            exp_x.append(math.exp(vector[i]))
        exp_x = (np.array(exp_x)).reshape(-1, 1)

        exp_minus_x = []
        for i in range(len(minus_vector)):
            exp_minus_x.append(math.exp(minus_vector[i]))
        exp_minus_x = (np.array(exp_minus_x)).reshape(-1, 1)

        res = (exp_x - exp_minus_x) / (exp_x + exp_minus_x)

        return res

    def hyperbolic_tangent_derivate(self, vector):
        ones = (np.ones(len(vector)))

        squared_vector = np.zeros(len(vector))
        for i in range(len(vector)):
            squared_vector[i] = math.sqrt(vector[i])

        res = ones - squared_vector

        return res

    def active_softmax(self, vector):
        exp_x = []
        for i in range(len(vector)):
            exp_x.append(math.exp(vector[i]))
        exp_x = (np.array(exp_x)).reshape(-1, 1)

        s = sum(exp_x)
        res = exp_x / s

        return res

    def softmax_derivative(self, ouput_batch):

        after_activation = []
        for i in range(len(ouput_batch[0])):
            activated = self.active_softmax(ouput_batch[:, i])
            activated = activated.reshape(-1)
            after_activation.append(activated)

        s = np.array(after_activation)
        # s = s.T

        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp1 = np.einsum('ij,jk->ijk',s,a)
        temp2 = np.einsum('ij,ik->ijk',s,s)

        res = temp1-temp2
        # res = res.T

        return res

        # OLD
        # s = self.active_softmax(vector)
        # n = len(vector)
        # jacobian_matrix = np.zeros((n, n))
        #
        # for i in range(n):
        #     for j in range(n):
        #         if i == j:
        #             jacobian_matrix[i, j] = s[i] * (1 - s[i])
        #         else:
        #             jacobian_matrix[i, j] = -s[i] * s[j]
        #
        # return jacobian_matrix

    def predict(self, inputs, weights, dropout_masks):
        outputs = inputs
        num_of_layers = len(self.nodes_in_layers)

        for i in range(1, num_of_layers):
            outputs = self.calculate_layer_output(weights[i - 1], outputs, i, dropout_masks)

        return outputs

    def calculate_h_layer_output(self, features, weights):
        output_vector = np.dot(weights[0], features)

        return output_vector

    def calculate_o_delta(self, results, expected_results):
        nodes_in_o_layer = len(expected_results)
        res = ((2 / nodes_in_o_layer) * np.subtract(results, expected_results))
        res = res / BATCH_SIZE

        return np.array(res)

    def calculate_h_delta(self, output_layer_delta, weights):
        # output_layer_delta = output_layer_delta.reshape(-1, 1)
        weights = np.array(weights)
        weights = weights.T

        res = np.dot(weights, output_layer_delta)
        return np.array(res)

    def calculate_delta_after_derivative(self, h_layer_delta, h_layer_outputs):
        after_derivative = np.array(self.relu_derivative(h_layer_outputs))
        res = np.multiply(h_layer_delta, after_derivative)

        return res

    def rescale_delta(self, layer_delta, prev_layer_output):
        prev_layer_output = prev_layer_output.T
        res = np.dot(layer_delta, prev_layer_output)

        return res

    def set_new_weights(self, all_weights, proper_layers_deltas):
        num_of_layers = len(all_weights)
        new_weights = []

        for i in range(0, num_of_layers):
            weights = all_weights[i] - LEARNING_RATE * proper_layers_deltas[i]
            new_weights.append(weights)

        return new_weights

    def count_deltas(self, outputs_of_all_series, expected_results, weights, outputs_of_layers, dropout_masks):
        proper_layers_deltas = []
        calculated_deltas = []

        o_layer_delta = self.calculate_o_delta(outputs_of_all_series, expected_results)
        calculated_deltas.append(o_layer_delta)

        h_layer_delta = self.calculate_h_delta(o_layer_delta, weights[1])
        h_layer_delta = np.array(self.calculate_delta_after_derivative(h_layer_delta, outputs_of_layers[1]))
        # WYGASZENIE!!!!!
        h_layer_delta = self.dropout(h_layer_delta, dropout_masks)
        calculated_deltas.append(h_layer_delta)


        input_index = len(calculated_deltas) - 1
        for i in range(0, len(calculated_deltas)):
            layer_delta = calculated_deltas[i]
            proper_delta = np.array(self.rescale_delta(layer_delta, outputs_of_layers[input_index]))
            if i == 0:
                proper_delta = proper_delta / BATCH_SIZE
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
        # old_weights = weights[0]

        for epoc in range(0, 1000):
            num_of_good_outputs = 0

            # if np.sum(np.array(old_weights[0]) - np.array(weights[0][0])) == 0:
            #     print(f"te same w epoce {epoc}\n")

            # res = np.array(old_weights[0]) - np.array(weights[0][0])
            # print(res)

            # NEW START
            dropout_masks = []
            for i in range(NUM_OF_SERIES):
                dropout_mask = np.random.binomial(1, 0.5, NODES_IN_H_LAYER)
                dropout_masks.append(dropout_mask)

            i = 0
            while i + BATCH_SIZE <= NUM_OF_SERIES:

                curr_inputs = features[:, i:(i + BATCH_SIZE)]
                total_outputs = np.array(self.predict(curr_inputs, weights, dropout_masks[i:(i + BATCH_SIZE)]))
                outputs_max = self.find_index_with_max_value(total_outputs)
                expected_outputs_max = self.find_index_with_max_value(expected_results[:, i:(i+BATCH_SIZE)])

                for j in range(BATCH_SIZE):
                    if expected_outputs_max[j] == outputs_max[j]:
                        num_of_good_outputs += 1

                all_outputs = [curr_inputs]
                for j in range(0, num_of_layers):
                    curr_inputs = (self.calculate_layer_output(weights[j], curr_inputs, j + 1, dropout_masks[i:(i + BATCH_SIZE)]))

                    all_outputs.append(curr_inputs)

                proper_layers_deltas = self.count_deltas(curr_inputs, expected_results[:, i:(i + BATCH_SIZE)],
                                                         weights, all_outputs, dropout_masks[i:(i + BATCH_SIZE)])
                weights = self.set_new_weights(weights, proper_layers_deltas)
                # res = np.sum(weights[0] - old_weights[0])
                # old_weights = weights

                i += BATCH_SIZE

                # if epoc % 100 == 0:
                #     print(f"{num_of_good_outputs}")

            if epoc % 100 == 0:
                accuracy = (num_of_good_outputs / NUM_OF_SERIES * 100)
                print(f"Accuracy for epoc {epoc}: {accuracy}%\n")

            self.save_to_file_new_data(FILENAME_WEIGHTS, weights)
            # NEW END

            # for i in range(0, 4):
            #     # ZMIANA 40->5
            #     dropout_mask = np.random.binomial(1, 0.5, 5)
            #     all_outputs = []
            #     curr_input = np.array(features[:, i]).reshape(-1, 1)
            #     expected_output = np.array(expected_results[:, i]).reshape(-1, 1)
            #     all_outputs.append(curr_input)
            #
            #     total_output = np.array(self.predict(curr_input, weights, dropout_mask)).reshape(-1, 1)
            #     output_max = self.find_index_with_max_value(total_output)
            #     expected_output_max = self.find_index_with_max_value(expected_output)
            #
            #     if expected_output_max == output_max:
            #         num_of_good_outputs += 1
            #
            #     error = sum(self.count_error(total_output, expected_output))
            #     sum_of_errors += error
            #
            #     for j in range(0, num_of_layers):
            #         curr_input = (self.calculate_layer_output(weights[j], curr_input, j + 1, dropout_mask)).reshape(-1,
            #                                                                                                         1)
            #
            #         all_outputs.append(curr_input)
            #
            #     proper_layers_deltas = self.count_deltas(curr_input, expected_output,
            #                                              weights, all_outputs, dropout_mask)
            #     weights = self.set_new_weights(weights, proper_layers_deltas)

            # if epoc % 10 == 0:
            #     accuracy = (num_of_good_outputs / 4 * 100)
            #     print(f"Accuracy for epoc {epoc}: {accuracy}%\n")

            # self.save_to_file_new_data(FILENAME_WEIGHTS, weights)

    def load_mnist_data(self):
        train_images = read_idx_file('MNIST_ORG/train-images.bin')
        train_labels = read_idx_file('MNIST_ORG/train-labels.bin')
        test_images = read_idx_file('MNIST_ORG/t10k-images.bin')
        test_labels = read_idx_file('MNIST_ORG/t10k-labels.bin')

        train_images = train_images / 255
        test_images = test_images / 255

        return train_images, train_labels, test_images, test_labels

    def find_index_with_max_value(self, outputs):
        indexes = []

        for i in range(BATCH_SIZE):
            max_val_index = 0

            for j in range(0, len(outputs)):
                if outputs[j][i] > outputs[max_val_index][i]:
                    max_val_index = j
            indexes.append(max_val_index)

        return indexes

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
    network = NeuralNetwork(784)
    np.random.seed(1)
    network.add_layer(NODES_IN_H_LAYER, activation_function=network.active_hyperbolic_tangent,
                      min_value=-0.01, max_value=0.01)
    network.add_layer(NODES_IN_O_LAYER, activation_function=network.active_softmax)
    train_images, train_labels, test_images, test_labels = network.load_mnist_data()

    # TRAIN IMAGES
    all_images_train = []
    for sublist in train_images:
        flat_list = []

        for item in sublist:
            flat_list.append(item)
        flat_list = np.concatenate(flat_list)
        flat_list = flat_list.reshape(-1)
        all_images_train.append(flat_list)
    features = (np.array(all_images_train)).T

    expected_res = np.zeros((10, len(train_labels)))  # 10 klas w MNIST
    for i, label in enumerate(train_labels):
        expected_res[label, i] = 1

    # TEST IMAGES
    all_images_test = []
    for sublist in test_images:
        flat_list = []

        for item in sublist:
            flat_list.append(item)
        flat_list = np.concatenate(flat_list)
        flat_list = flat_list.reshape(-1)
        all_images_test.append(flat_list)
    features_test = (np.array(all_images_test)).T

    expected_res_test = np.zeros((10, len(test_labels)))  # 10 klas w MNIST
    for i, label in enumerate(test_labels):
        expected_res_test[label, i] = 1

    # network.active_sigmoid(features[150:156, 0])
    network.fit(expected_res, features)

    # num_of_good_outputs = network.test_neural_network(features_test, expected_res_test, FILENAME_WEIGHTS)
    # print(f"Accuracy of good outputs in test: {num_of_good_outputs / 1000 * 100}%\n")

if __name__ == "__main__":
    main()

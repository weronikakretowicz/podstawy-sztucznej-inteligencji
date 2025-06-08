import numpy as np

LEARNING_RATE = 0.01


def calculate_h_layer_output(features, weights):
    output_vector = np.dot(weights[0], features)
    size = len(output_vector)

    for val in range(0, size):
        output_vector[val] = max(0, output_vector[val])

    return output_vector.T


def predict(features, weights):
    num_of_layers = len(weights)
    index = 0
    input_vector = features

    while (index < num_of_layers):
        output_vector = np.dot(weights[index], input_vector)

        if (index == 0):
            for val in range(0, len(output_vector)):
                output_vector[val] = max(0, output_vector[val])

        input_vector = output_vector.T
        index += 1

    return input_vector


def calculate_o_delta(res, expected_result):
    nodes_in_o_layer = len(expected_result)
    res = (2 / nodes_in_o_layer * np.subtract(res, expected_result))

    return np.array(res)


def calculate_h_delta(output_layer_delta, weights):
    output_layer_delta = np.array(output_layer_delta).T
    weights = weights.T

    res = np.dot(weights, output_layer_delta)
    return np.array(res)


def calculate_der_relu(vector):
    data = [1 if value > 0 else 0 for value in vector]
    return np.array(data)


def multiply_vectors(h_layer_delta, vector):
    for i in range(0, len(vector)):
        h_layer_delta[i] = h_layer_delta[i] * vector[i]

    return h_layer_delta


def calculate_delta_after_derivative(h_layer_delta, h_layer_output):
    val = calculate_der_relu(h_layer_output)

    return multiply_vectors(h_layer_delta, val)


def rescale_delta(layer_delta, prev_layer_output):
    return np.multiply(layer_delta, prev_layer_output)


def set_new_weights(all_weights, proper_layers_deltas):
    num_of_layers = len(all_weights)
    new_weights = []

    for i in range(0, num_of_layers):
        weights = all_weights[i] - LEARNING_RATE * proper_layers_deltas[i]
        new_weights.append(weights)

    return new_weights

def print_result(num_of_epoch, num_of_series, res):
    if (num_of_series == 0):
        print(f"Wynik w {num_of_epoch + 1} epoce:\n")

    print(f"wyjście sieci po {num_of_series + 1} serii: {np.round(res, 6)}\n")
    if (num_of_series == 3):
        print("\n")

def count_error(my_output, proper_output):
    return np.subtract(my_output, proper_output) ** 2

def main():
    features = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    expected_results = np.array([[0.1, 0.5, 0.1, 0.7], [1, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])

    weights_h = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0], [0, 0.7, 0.1], [0.2, 0.4, 0], [-0.3, 0.5, 0.1]])
    weights_o = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
    weights = [weights_h, weights_o]
    proper_layers_deltas = []

    num_of_series = len(features[0])
    for i in range(0, 50):
        index = 0

        while (index < num_of_series):
            res = predict(features[:, index], weights)
            # error = sum(count_error(res, expected_results[:, index]))
            # print(error)

            if (i == 0 or i == 49):
                print_result(i, index, res)

            o_layer_delta = calculate_o_delta(res, expected_results[:, index])
            h_layer_delta = calculate_h_delta(o_layer_delta, weights[1])
            h_layer_delta = calculate_delta_after_derivative(h_layer_delta,
                                                             calculate_h_layer_output(features[:, index], weights))

            o_layer_delta = o_layer_delta.reshape(-1, 1)
            proper_o_layer_delta = rescale_delta(o_layer_delta, calculate_h_layer_output(features[:, index], weights))

            h_layer_delta = h_layer_delta.reshape(-1, 1)
            curr_input = (features[:, index]).reshape(1, -1)
            proper_h_layer_delta = rescale_delta(h_layer_delta, curr_input)
            proper_layers_deltas.append(proper_h_layer_delta)
            proper_layers_deltas.append(proper_o_layer_delta)

            weights = set_new_weights(weights, proper_layers_deltas)
            proper_layers_deltas.clear()

            index += 1


if __name__ == "__main__":
    main()

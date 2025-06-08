import numpy as np


def calculate_h_layer_output(features, weights):
    output_vector = np.dot(weights[0], features)

    for val in range(0, len(features)):
        output_vector[val] = max(0, output_vector[val])

    return output_vector.T


def predict(features, weights):
    num_of_layers = len(weights)
    index = 0
    input_vector = features

    while (index < num_of_layers):
        output_vector = np.dot(weights[index], input_vector)

        for val in range(0, len(output_vector)):
            output_vector[val] = max(0, output_vector[val])
        input_vector = output_vector.T

        index += 1

    return input_vector

def main():
    features = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    weights_h = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0], [0, 0.7, 0.1], [0.2, 0.4, 0], [-0.3, 0.5, 0.1]])
    weights_o = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
    weights = []
    weights.append(weights_h)
    weights.append(weights_o)

    num_of_series = len(features[0])
    index = 0
    while (index < num_of_series):
        res = predict(features[:, index], weights)
        print(f"Wyście po {index} serii: {res}\n")
        index += 1

if __name__ == "__main__":
    main()

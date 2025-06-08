import numpy as np

def check_if_properties_are_correct(matrix, vector):
    columns = len(matrix[0])
    rows = len(vector)

    if rows != columns:
        return 1
    else:
        return 0

def count_layer_output(matrix, vector):
    if check_if_properties_are_correct(matrix, vector):
        return 1
    else:
        return np.dot(matrix, vector)

def deep_neural_network(features, list_of_weights):
    last_layer_outcome = features
    index_layer = 0

    while (index_layer < len(list_of_weights)):
        res = count_layer_output(list_of_weights[index_layer], last_layer_outcome)
        if (type(res) == int):
            return 1
        else:
            last_layer_outcome = res

        index_layer += 1

    return last_layer_outcome

def main():
    features = np.array([0.5, 0.75, 0.1]).T
    hidden_layer_weights = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
    output_layer_weights = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])

    all_weights = []
    all_weights.append(hidden_layer_weights)
    all_weights.append(output_layer_weights)

    output = deep_neural_network(features, all_weights)
    if (type(output) == int):
        print(f"Incorrect input dimensions!\n")
    else:
        print(f"Output vector: {output}\n")

if __name__ == "__main__":
    main()

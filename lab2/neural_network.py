import numpy as np

LEARNING_RATE = 0.01


def calculate_layer_output(features, weights):
    columns = len(weights[0])
    rows = len(features)

    if rows != columns:
        return 1
    else:
        res = np.dot(weights, features)

    return np.array(res).T


def calculate_error(my_output, proper_output):
    return (my_output - proper_output) ** 2


def set_new_weights(features, weights, my_output, proper_output):
    delta = (2 * (1 / len(my_output))) * np.outer(np.subtract(my_output, proper_output), features)
    weights = np.subtract(weights, LEARNING_RATE * delta)

    return weights


def main():
    features = np.array([
        [0.5, 0.1, 0.2, 0.8],
        [0.75, 0.3, 0.1, 0.9],
        [0.1, 0.7, 0.6, 0.2]
    ])
    weights = np.array([
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ])
    proper_output = np.array([[
        0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2],
        [0.0, 0.3, 0.9, -0.1],
        [-0.1, 0.7, 0.1, 0.8]
    ])
    my_output = []
    counter = 0
    num_of_series = len(features[0])

    while (counter < 1000):
        sum_of_errors = 0
        counter += 1

        for i in range(0, num_of_series):
            my_output = calculate_layer_output(features[:, i], weights)
            error = sum(calculate_error(my_output, proper_output[:, i]))
            sum_of_errors += error
            weights = set_new_weights(features[:, i], weights, my_output, proper_output[:, i])

        res = (1 / len(my_output)) * sum_of_errors
        print(f"{counter} Error: {round(res, 6)}\n")

    print(f"Output :\n {my_output}\n")


if __name__ == "__main__":
    main()

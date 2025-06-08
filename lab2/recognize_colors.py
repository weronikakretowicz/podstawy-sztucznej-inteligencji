import numpy as np
import csv
LEARNING_RATE = 0.01

def are_dimensions_correct(features, weights):
    columns = len(weights[0])
    rows = len(features)

    if rows != columns:
        return 1
    else:
        return 0

def count_output(features, weights):
    if are_dimensions_correct(features, weights):
        return 1
    else:
        return np.array(np.dot(weights, features)).T

def count_error(my_output, proper_output):
    return np.subtract(my_output, proper_output) ** 2

def set_new_weights(features, weights, my_output, proper_output):
    delta = (2 * (1 / len(my_output))) * np.outer(np.subtract(my_output, proper_output), features)
    weights = np.subtract(weights, LEARNING_RATE * delta)

    return weights


def train_neural_network(file_name):
    features = []
    proper_output = []
    weights = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0]])

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for i in range(0, 50):
            for row in csv_reader:
                features.clear()

                features.append(float(row[0]))
                features.append(float(row[1]))
                features.append(float(row[2]))

                if row[3] == '1':
                    proper_output = [1, 0, 0, 0]
                elif row[3] == '2':
                    proper_output = [0, 1, 0, 0]
                elif row[3] == '3':
                    proper_output = [0, 0, 1, 0]
                elif row[3] == '4':
                    proper_output = [0, 0, 0, 1]

                my_output = count_output(np.array(features).T, weights)
                if type(my_output) == int:
                    return 1

                weights = set_new_weights(np.array(features).T, weights, my_output, np.array(proper_output).T)

            csv_file.seek(0)

        csv_file.close()

    return weights

def find_equal_index(vector, value):
    for i in range(0, len(vector)):
        if vector[i] == value:
            return i

def is_output_correct(my_output, proper_output):
    max_value = max(my_output)
    my_output_index = find_equal_index(my_output, max_value)
    proper_output_index = find_equal_index(proper_output, 1)

    return 1 if my_output_index == proper_output_index else 0

def test_neural_network(file_name, weights):
    features = []
    proper_output = []
    num_of_wrong_outputs = 0

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader, None)

        for row in csv_reader:
            features.clear()

            features.append(float(row[0]))
            features.append(float(row[1]))
            features.append(float(row[2]))

            if row[3] == '1':
                proper_output = [1, 0, 0, 0]
            elif row[3] == '2':
                proper_output = [0, 1, 0, 0]
            elif row[3] == '3':
                proper_output = [0, 0, 1, 0]
            elif row[3] == '4':
                proper_output = [0, 0, 0, 1]

            my_output = count_output(np.array(features).T, weights)
            if is_output_correct(my_output, np.array(proper_output).T) == 0:
                num_of_wrong_outputs += 1

    csv_file.close()

    return num_of_wrong_outputs

def main():
    weights = train_neural_network("training.csv")
    if type(weights) == int:
        print(f"Incorrect dimensions!\n")
        return

    num_of_wrong_outputs = test_neural_network("test.csv", weights)
    print(f"Number of wrong answers: {num_of_wrong_outputs}\n")

if __name__ == "__main__":
    main()
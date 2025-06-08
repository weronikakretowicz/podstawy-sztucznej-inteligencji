import numpy as np

def are_properties_correct(features, layer_weights):
    if len(features) == len(layer_weights):
        return 1
    else:
        return 0

def count_layer_output(features, layer_weights):
    return features * layer_weights

def main():
    input = 2
    weight = 0.5
    proper_output = 0.8
    learning_rate = 0.1
    counter = 0

    while (counter != 20):
        predicted_output = count_layer_output(input, weight)
        error = (predicted_output - proper_output)**2
        delta = 2*(predicted_output - proper_output)*input
        weight = weight - learning_rate*delta

        if counter == 4:
            print(f"Error: {error}\n")
            print(f"Output {predicted_output}\n")

        counter += 1


if __name__ == "__main__":
    main()

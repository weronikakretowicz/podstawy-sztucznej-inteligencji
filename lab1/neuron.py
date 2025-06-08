import numpy as np

def single_neuron(features, weights, bias):
    res = features * weights + bias
    return res

"""
                      0   if x1 < 0
heaviside(x1, x2) =  x2   if x1 == 0
                      1   if x1 > 0
"""

def main():
    np.random.seed(1)
    features = np.array(np.random.randint(101, size=5))
    print(f"Features: {features}\n")

    weights = np.array(np.random.random(5)).T
    print(f"Weights: {weights}\n")
    bias = round(np.random.random(), 2)
    print(f"Bias: {bias}\n")

    result = single_neuron(features, weights, bias)
    print(f"Wynik: {result}\n")

if __name__ == "__main__":
    main()


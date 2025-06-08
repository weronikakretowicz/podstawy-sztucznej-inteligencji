import csv
import struct
import math
import numpy as np

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

class ConfNet :
    def __init__(self, kernel_stride, num_of_kernels, kernel_size, num_of_nodes_in_last_layer, learning_rate, image_size, pooling_stride, pooling_mask_size):
        self.num_of_kernels = num_of_kernels
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kernel_stride = kernel_stride
        self.image_size = image_size
        self.num_of_nodes_in_last_layer = num_of_nodes_in_last_layer
        self.conv_output = []
        self.fc_output = None
        self.fragments_of_image = []
        self.conv_kernel_reshaped = None
        self.pooling_stride = pooling_stride
        self.pooling_mask_size = pooling_mask_size
        self.pooled_output = None
        self.max_indices = None
        self.conv_kernel, self.fc_weights = self.initialize_weights()

    def initialize_weights(self):
        conv_kernel = np.random.uniform(-0.01, 0.01, size=(self.num_of_kernels, self.kernel_size, self.kernel_size))

        num_of_image_fragments = round(((self.image_size - self.kernel_size)/self.kernel_stride + 1) * \
                                       ((self.image_size - self.kernel_size)/self.kernel_stride + 1))
        num_of_flattened_conv_outputs = self.num_of_kernels * num_of_image_fragments // (self.pooling_mask_size * 2)
        fc_weights = np.random.uniform(-0.1, 0.1, size=(self.num_of_nodes_in_last_layer, num_of_flattened_conv_outputs))

        return conv_kernel, fc_weights

    def load_mnist_data(self):
        train_images = read_idx_file('MNIST_ORG/train-images.bin')
        train_labels = read_idx_file('MNIST_ORG/train-labels.bin')
        test_images = read_idx_file('MNIST_ORG/t10k-images.bin')
        test_labels = read_idx_file('MNIST_ORG/t10k-labels.bin')

        train_images = train_images / 255
        test_images = test_images / 255

        return train_images, train_labels, test_images, test_labels

    def activate_relu(self, matrix):
        return np.maximum(0, matrix)

    def extract_and_transform_image(self, image):
        fragments_of_image = []
        image_s = round((self.image_size - self.kernel_size)/self.kernel_stride + 1)

        for i in range(image_s):
            for j in range(image_s):
                fragment = image[i*self.kernel_stride: i*self.kernel_stride+self.kernel_size, j*self.kernel_stride: j*self.kernel_stride+self.kernel_size]

                fragment_vector = fragment.flatten()

                fragments_of_image.append(fragment_vector)

        return np.array(fragments_of_image)

    def transform_kernels(self):
        matrix_of_kernels = []

        for i in range(self.num_of_kernels):
            matrix_of_kernels.append((self.conv_kernel[i]).flatten())

        return (np.array(matrix_of_kernels)).T

    def max_pooling(self):
        shape_of_conv_output = np.array(self.conv_output).shape #676x16
        new_weight = round((shape_of_conv_output[0] - self.pooling_mask_size)/self.pooling_stride + 1)
        new_height = round((shape_of_conv_output[1] - self.pooling_mask_size)/self.pooling_stride + 1)
        pooled_output = np.zeros((new_weight, new_height), dtype=float)
        max_indices = np.zeros_like(self.conv_output)

        for i in range(0, shape_of_conv_output[0], self.pooling_stride):
            for j in range(0, shape_of_conv_output[1], self.pooling_stride):
                pool_fragment = self.conv_output[i:i+self.pooling_stride, j:j+self.pooling_stride]
                pooled_output[i//self.pooling_stride, j//self.pooling_stride] = np.max(pool_fragment)

                max_index = np.unravel_index(np.argmax(pool_fragment, axis=None), pool_fragment.shape)
                max_indices[i + max_index[0], j + max_index[1]] = 1

        self.max_indices = max_indices
        self.pooled_output = pooled_output.flatten()

    def perform_convolution(self, image):
        self.fragments_of_image = self.extract_and_transform_image(image) #676x9
        self.conv_output = self.activate_relu(np.dot(self.fragments_of_image, self.conv_kernel_reshaped))

    def perform_fully_connected_l(self):
        self.fc_output = np.dot(self.fc_weights, self.pooled_output)

    def forward_propagation(self, image):
        self.perform_convolution(image)
        self.max_pooling()
        self.perform_fully_connected_l()

    def fill_mask_with_vector(self, conv_delta_vec):
        proper_conv_delta = np.zeros_like(self.conv_output)
        conv_shape = np.array(self.conv_output).shape
        vec_i = 0

        for i in range(0, conv_shape[0], self.pooling_stride):
            for j in range(0, conv_shape[1], self.pooling_stride):
                proper_conv_delta[i:i+self.pooling_stride, j:j+self.pooling_stride] = conv_delta_vec[vec_i]
                vec_i += 1

        return proper_conv_delta

    def back_propagation(self, expected_res):
        fc_delta = self.calculate_fc_delta(expected_res)
        conv_delta = self.calculate_conv_delta(fc_delta)
        conv_delta = self.calculate_delta_after_derivative(conv_delta)

        conv_delta = self.fill_mask_with_vector(conv_delta)
        conv_delta = self.max_indices * conv_delta

        proper_deltas = [np.dot(conv_delta.T, self.fragments_of_image),
                         fc_delta.reshape(-1, 1) * np.array(self.pooled_output).T]

        return proper_deltas

    def set_new_weights(self, proper_deltas):
        self.conv_kernel_reshaped = (self.conv_kernel_reshaped.T - self.learning_rate * proper_deltas[0]).T
        self.fc_weights = self.fc_weights - self.learning_rate * proper_deltas[1]

    def calculate_fc_delta(self, expected_results):
        nodes_in_o_layer = len(expected_results)
        res = ((2 / nodes_in_o_layer) * np.subtract(self.fc_output, expected_results))

        return np.array(res)

    def calculate_conv_delta(self, fc_layer_delta):
        weights = self.fc_weights.T
        res = np.dot(weights, fc_layer_delta)

        return np.array(res)

    def calculate_delta_after_derivative(self, layer_delta):
        after_derivative = np.array([1 if value > 0 else 0 for value in self.pooled_output])
        res = np.multiply(layer_delta, after_derivative)

        return res

    def find_index_with_max_value(self, my_output):
        max_val_index = 0

        for j in range(0, len(my_output)):
            if my_output[j] > my_output[max_val_index]:
                max_val_index = j

        return max_val_index

    def fit(self, expected_results, images):
        self.conv_kernel_reshaped = self.transform_kernels() #9x16

        for epoc in range(0, 30):
            num_of_series = len(images)
            num_of_good_outputs = 0

            for i in range(0, 1000):
                self.forward_propagation(images[i])
                output_max = self.find_index_with_max_value(self.fc_output)
                expected_output_max = self.find_index_with_max_value(expected_results[:, i])

                if expected_output_max == output_max:
                    num_of_good_outputs += 1

                proper_deltas = self.back_propagation(expected_results[:, i])
                self.set_new_weights(proper_deltas)

            if epoc % 10 == 0:
                print(f"Accuracy for epoc {epoc}: {(num_of_good_outputs / num_of_series * 100)}%\n")

    def test_neural_network(self, images, expected_output):
        num_of_good_outputs = 0

        for i in range(0, len(images)):
            self.forward_propagation(images[i])
            output_max = self.find_index_with_max_value(self.fc_output)
            expected_output_max = self.find_index_with_max_value(expected_output[:, i])

            if expected_output_max == output_max:
                num_of_good_outputs += 1

        return num_of_good_outputs


def main():
    np.random.seed(1)
    # kernel_stride, num_of_kernels, kernel_size, num_of_nodes_in_last_layer, learning_rate, image_size, pooling_stride, pooling_mask_size
    network = ConfNet(1, 16, 3, 10, 0.1, 28, 2, 2)
    train_images, train_labels, test_images, test_labels = network.load_mnist_data()
    expected_res = np.zeros((10, len(train_labels)))  # 10 klas w MNIST
    for i, label in enumerate(train_labels):
        expected_res[label, i] = 1

    expected_res_test = np.zeros((10, len(test_labels)))  # 10 klas w MNIST
    for i, label in enumerate(test_labels):
        expected_res_test[label, i] = 1

    network.fit(expected_res[:, :1000], train_images[:1000])
    num_of_good_outputs = network.test_neural_network(test_images[:10000], expected_res_test[:, :10000])
    print(f"Accuracy for 10 000 test images: {(num_of_good_outputs / 10000 * 100)}%\n")


if __name__ == "__main__":
    main()
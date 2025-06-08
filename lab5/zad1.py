import numpy as np

def convolution(image, kernel, stride):
    padded_image = np.pad(image, pad_width=0, mode='constant', constant_values=0)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    output_image = np.zeros((output_height, output_width))

    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]

            output_image[i//stride, j//stride] = np.sum(roi * kernel)

    return output_image

input_image = np.array([[1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0]])

filter_kernel = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 1]])

stride = 1
result = convolution(input_image, filter_kernel, stride)
print(result)




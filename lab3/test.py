import numpy as np
import struct
from array import array

row = '-0.165955990594852 0.4406489868843162 -0.9997712503653102 -0.39533485473632046 -0.7064882183657739'
clean_line = row.strip()

float_numbers = []or num in clean_line.split():
    float_numbers.append(float(num))

curr_row = np.array(float_numbers)

# curr_matrix.append([909, 93, 54, 646, 77])
curr_matrix = []
curr_matrix.append(curr_row)
print(curr_matrix)
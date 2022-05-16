"""
Author: Artin

"""

import numpy as np


def mask_image(image_array, offset, spacing):

    # Check if image_array is instance of np.ndarray otherwise raise TypeError
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array is not a numpy array!")

    # Check if image_array is a 3D array with 3 values in each cell otherwise raise NotImplementedError
    image_array_shape = image_array.shape
    if image_array.ndim != 3 or image_array_shape[2] != 3:
        raise NotImplementedError("image_array is not a 3D array or the size of the 3rd dimension is not equal to 3")

    # Convert offset and spacing values to int otherwise numpy will raise ValueError
    offset = np.array(offset, dtype=np.int)
    spacing = np.array(spacing, dtype=np.int)

    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    # Calculate number of known pixels
    n_known_px = np.ceil((image_height - offset[1]) / spacing[1]) * np.ceil((image_width - offset[0]) / spacing[0])

    # Check if offset and spacing values are between 0 and 32 and 2 and 8 respectively and number of the remaining
    # known image pixels is greater than 144 otherwise raise ValueError
    if offset[0] < 0 or offset[1] < 0 or offset[0] > 32 or offset[1] > 32 or spacing[0] < 2 or spacing[1] < 2 or \
            spacing[0] > 8 or spacing[1] > 8 or n_known_px < 144:
        raise ValueError("There is a problem with offset or spacing parameters")

    # Create input_array and known_array with zero values and with same shape as image_array
    input_array = np.zeros_like(image_array)
    known_array = np.zeros_like(image_array)

    # Fill known_array by setting known pixels to 1 and input_value to actual pixel values
    for m in range(offset[1], image_height, spacing[1]):
        for n in range(offset[0], image_width, spacing[0]):
            known_array[m][n] = [1, 1, 1]
            input_array[m][n] = image_array[m][n]

    # Make a mask from image_array with containing only pixel values of parts which were set to zero
    masked_array = image_array[known_array < 1]

    # Make target_array and fill with R followed by G followed by B values from masked_array
    target_array = np.array([], dtype=masked_array.dtype)
    target_array = np.append(target_array, masked_array[::3])
    target_array = np.append(target_array, masked_array[1::3])
    target_array = np.append(target_array, masked_array[2::3])

    # Return input_array and known_array in shape of (3, M, N)
    return np.transpose(input_array, (2, 0, 1)), np.transpose(known_array, (2, 0, 1)), target_array

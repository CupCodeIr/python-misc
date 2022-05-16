"""
Author: Artin

"""
import glob
import numpy as np
import os

from PIL import Image


class ImageStandardizer:

    def __init__(self, input_dir):
        # Getting absolute path of input and output directory
        input_dir_abs = os.path.abspath(input_dir)
        # Getting all files ending in .jpg inside input_dir recursively sorted base file name alphabetically in
        # ascending order
        files = sorted(glob.glob(os.path.join(input_dir_abs, "**", "*.jpg"), recursive=True))
        # Raise error if no files found
        if len(files) < 1:
            raise ValueError("No .jpg file exists in the directory")
        # Setting files path to class-wide self.files variable
        self.files = files
        # Define class-wide variables
        self.mean = None
        self.std = None

    def analyze_images(self):
        #  Define array to store means and standard deviations of each RGB image
        rgb_means = []
        rgb_stds = []
        # Loop through files path
        for file in self.files:
            # Open file as Image
            image = Image.open(file)
            # Convert image to array
            image_array = np.array(image)
            # Calculate Mean of each image and store inside rgb_means
            rgb_means.append(np.mean(image_array, axis=(0, 1)))
            # Calculate Standard Deviation of each image and store inside rgb_stds
            rgb_stds.append(np.std(image_array, axis=(0, 1)))
        # Store average of Means
        self.mean = np.average(rgb_means, axis=0)
        # Store average of Standard Deviations
        self.std = np.average(rgb_stds, axis=0)
        # Return global mean and std as a tuple
        return self.mean, self.std

    def get_standardized_images(self):
        # Raise ValueError if global mean and std were not calculated
        if self.mean is None or self.std is None:
            raise ValueError("Mean or STD is None")
        for image_path in self.files:
            image = Image.open(image_path)
            # Load image as a numpy array with data type np.float32
            image_array = np.array(image, dtype=np.float32)
            # Normalizing image pixel data by first subtracting global mean from each pixel data
            # and then diving by global std
            np.subtract(image_array, self.mean, out=image_array)
            np.divide(image_array, self.std, out=image_array)
            yield image_array

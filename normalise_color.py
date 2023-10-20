"""
This file contains the function that performs global color normalization on an image.
"""
import numpy as np


def normalize_global_color(image):
    """
    Performs global color normalization on an image.
    :param image: A numpy array representing the image.
    :return: A numpy array representing the normalized image.
    """

    # Compute the mean and standard deviation of each color channel.
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    # Normalize the image.
    normalized_image = (image - mean) / std

    return normalized_image

"""
This file contains the function that performs global color normalization on an image.
"""
import numpy as np
import cv2 as cv


def normalize_global_color_type(image, T=np.float32):
    """
    Performs global color normalization on an image.
    :param image: A numpy array representing the image.
    :param T: The type of the image, for now only floats and doubles are supported.
    :return: normalized_image (T): A numpy array representing the normalized image
    """
    # Compute the mean and standard deviation of each color channel.
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    # Normalize the image.
    normalized_image = ((image - mean) / std).astype(T)

    # convert to 0-255, CV_8U
    normalized_image = cv.normalize(normalized_image, normalized_image, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                    dtype=cv.CV_8U)
    return normalized_image

"""
This file contains the function that performs global color normalization on an image.
"""
import numpy as np
import cv2 as cv
from camera_calibration import rectify_images
from remove_background import get_foreground_mask_HSV


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


def preprocess(images, calibration_data=None, suffix=None, display=False):
    """
    Preprocesses the images by normalizing the global color and removing the background. Also performs stereo rectification.
    Args:
        display: Whether to display the images.
        images: A list of two images, unrectified.
        calibration_data: The calibration data dictionary.
        suffix: The suffix of the calibration data to use.

    Returns:

    """
    mask = np.array([None, None])
    for i in [0, 1]:
        # Make Foreground mask
        mask[i] = get_foreground_mask_HSV(images[i])

        # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
        images[i] = normalize_global_color_type(images[i])
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    rect_images = rectify_images(images[0], images[1], calibration_data[f'map1x{suffix}'],
                                 calibration_data[f'map1y{suffix}'],
                                 calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'],
                                 calibration_data[f'ROI1{suffix}'], calibration_data[f'ROI2{suffix}'], suffix)
    rect_mask = rectify_images(mask[0], mask[1], calibration_data[f'map1x{suffix}'],
                               calibration_data[f'map1y{suffix}'],
                               calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'],
                               calibration_data[f'ROI1{suffix}'], calibration_data[f'ROI2{suffix}'], suffix)
    if display:  # Display the images
        # Apply the mask to the images as a translucent red overlay
        im1 = rect_images[0].copy()
        im2 = rect_images[1].copy()
        im1[rect_mask[0] != 255] = [0, 0, 255]
        im2[rect_mask[1] != 255] = [0, 0, 255]
        im1 = cv.addWeighted(im1, 0.5, rect_images[0], 0.5, 0)
        im2 = cv.addWeighted(im2, 0.5, rect_images[1], 0.5, 0)
        # Display the masked images
        cv.imshow(f"Left{suffix}", im1)
        cv.imshow(f"Right{suffix}", im2)
        cv.waitKey(0)
    return rect_images, rect_mask

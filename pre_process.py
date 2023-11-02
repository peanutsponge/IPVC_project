"""
This file contains the preprocessing function along with one that performs global color normalization on an image.
"""
import numpy as np
import cv2 as cv
from camera_calibration import rectify_images
from remove_background import get_foreground_mask_HSV, get_foreground_mask_HSV_interactively


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
    image = ((image - mean) / std).astype(T)

    # convert to 0-255, CV_8U
    image = cv.normalize(image, image, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)
    return image


def preprocess(images, calibration_data=None, suffix=None, save_path='output', display=False):
    """
    Preprocesses the images by normalizing the global color and removing the background. Also performs stereo rectification.
    Args:
        save_path:
        display: Whether to display the images.
        images: A list of two images, unrectified.
        calibration_data: The calibration data dictionary.
        suffix: The suffix of the calibration data to use.

    Returns:

    """
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    images = rectify_images(images[0], images[1], calibration_data[f'map1x{suffix}'],
                            calibration_data[f'map1y{suffix}'],
                            calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'],
                            calibration_data[f'ROI1{suffix}'], calibration_data[f'ROI2{suffix}'], suffix)
    mask = np.ones_like(images)[:, :, :, 0] * 255
    for i in [0, 1]:
        # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
        images[i] = normalize_global_color_type(images[i])

        # get_foreground_mask_HSV_interactively(images[i]) # Uncomment this line to interactively select the HSV values
        # Make Foreground mask
        mask[i] = get_foreground_mask_HSV(images[i],
                                          cleaning_amount=50, closing_amount=5, fill_holes=True,
                                          v_min=40, v_max=255,  # To get rid of hair mainly
                                          s_min=0, s_max=50,  # To get rid of clothes
                                          h_min=100, h_max=255,  # To isolate the skin
                                          hue_shift=200)

    # Apply the mask to the images as a translucent red overlay
    im1 = images[0].copy()
    im2 = images[1].copy()

    im1[mask[0] != 255] = [0, 0, 255]
    im2[mask[1] != 255] = [0, 0, 255]

    im1 = cv.addWeighted(im1, 0.5, images[0], 0.5, 0)
    im2 = cv.addWeighted(im2, 0.5, images[1], 0.5, 0)

    # Save the images
    cv.imwrite(f"{save_path}{suffix}_Left.jpg", im1)
    cv.imwrite(f"{save_path}{suffix}_Right.jpg", im2)

    if display:  # Display the images
        # Display the masked images
        cv.imshow(f"Left{suffix}", im1)
        cv.imshow(f"Right{suffix}", im2)
        cv.waitKey(0)

    return images, mask

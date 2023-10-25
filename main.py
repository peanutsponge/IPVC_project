"""
Main file that runs the entire pipeline
"""
import os
import cv2 as cv
import numpy as np

from remove_background import get_foreground_mask
from get_images import getTriplet
from mesh import generate_mesh
from normalise_color import normalize_global_color
from camera_calibration import get_calibration_data_from_file, calibrate_3_cameras_to_file, apply_lens_correction, \
    rectify_images


def preprocess(images):
    mask = [None, None]
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    for i in [0, 1]:
        # Make Foreground mask
        mask[i] = get_foreground_mask(images[i])

        # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
        images[i] = normalize_global_color(images[i])
    return images, mask


# Only execute to generate the calibration data file
# calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

# Camera id mappings
camera_id = {0: "left", 1: "middle", 2: "right"}

triplet = getTriplet(1, 0)  # In final version we'll loop over these

# Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
rect_images_LM = rectify_images(triplet[0], triplet[1], calibration_data['map1x_lm'], calibration_data['map1y_lm'],
                                calibration_data['map2x_lm'], calibration_data['map2y_lm'])
rect_images_MR = rectify_images(triplet[1], triplet[2], calibration_data['map1x_mr'], calibration_data['map1y_mr'],
                                calibration_data['map2x_mr'], calibration_data['map2y_mr'])

# Preprocess the images
images_LM, mask_LM = preprocess(rect_images_LM)
images_MR, mask_MR = preprocess(rect_images_MR)

# Display the images
# cv.imshow('Left', triplet[0])
# cv.imshow('Middle', triplet[1])
# cv.imshow('Right', triplet[2])
# cv.waitKey(0)

# Generate two meshes
mesh_LM = generate_mesh(images_LM, mask_LM)
mesh_MR = generate_mesh(images_MR, mask_MR)

# Merge the meshes using the ICP algorithm (iterated closest points)


# Save the mesh to a file

# Plot the mesh

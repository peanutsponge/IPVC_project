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
from camera_calibration import get_calibration_data_from_file, calibrate_3_cameras_to_file, apply_lens_correction, rectify_images

# Only execute to generate the calibration data file
#calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

# Camera id mappings
camera_id = {0: "left", 1: "middle", 2: "right"}

mask = [None, None, None]
triplet = getTriplet(1, 0)  # In final version we'll loop over these
for i, im in enumerate(triplet):
    # Make Foreground mask
    mask[i] = get_foreground_mask(im)

    # Apply lens correction, not necessary here! is performed in rectify_images
    # im = apply_lens_correction(im, camera_id[i], calibration_data)
    #show the image

    # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
    im = normalize_global_color(im)

    # Save the image to the triplet
    triplet[i] = im

#TODO: Where to put this? In the loop or after the loop? Doesn't really work with the loop
# Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
rect_images_LM = (triplet[0], triplet[1], calibration_data['map1x_lm'], calibration_data['map1y_lm'], calibration_data['map2x_lm'], calibration_data['map2y_lm'])
rect_images_MR = (triplet[1], triplet[2], calibration_data['map1x_mr'], calibration_data['map1y_mr'], calibration_data['map2x_mr'], calibration_data['map2y_mr'])

# Display the images
cv.imshow('Left', triplet[0])
cv.imshow('Middle', triplet[1])
cv.imshow('Right', triplet[2])
cv.waitKey(0)

# Generate two meshes
mesh_LM = generate_mesh(im[0], im[1], mask[0], mask[1])
mesh_MR = generate_mesh(im[1], im[2], mask[1], mask[2])

# Merge the meshes using the ICP algorithm (iterated closest points)


# Save the mesh to a file

# Plot the mesh

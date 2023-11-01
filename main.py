"""
Main file that runs the entire pipeline
"""
import os
import cv2 as cv
import numpy as np

from remove_background import get_foreground_mask, get_foreground_mask_HSV
from get_images import getTriplet
from mesh import generate_point_cloud, create_point_cloud_file, create_mesh
from normalise_color import normalize_global_color, normalize_global_color_type
from camera_calibration import get_calibration_data_from_file, calibrate_3_cameras_to_file, apply_lens_correction, \
    rectify_images

#* Settings
displayImagesWhileRunning = False

def preprocess(images, suffix=None):
    mask = [None, None]
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    for i in [0, 1]:
        # Make Foreground mask
        mask[i] = get_foreground_mask_HSV(images[i])

        # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
        images[i] = normalize_global_color_type(images[i])
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    rect_images = rectify_images(images[0], images[1], calibration_data[f'map1x{suffix}'],
                                 calibration_data[f'map1y{suffix}'],
                                 calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'])
    rect_mask = rectify_images(mask[0], mask[1], calibration_data[f'map1x{suffix}'],
                               calibration_data[f'map1y{suffix}'],
                               calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'])
    return rect_images, rect_mask


# Only execute to generate the calibration data file
# calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

# Camera id mappings
camera_id = {0: "left", 1: "middle", 2: "right"}

triplet = getTriplet(1, 0)  # In final version we'll loop over these

# Preprocess the images
images_LM, mask_LM = preprocess([triplet[0], triplet[1]], "_lm")
images_MR, mask_MR = preprocess([triplet[1], triplet[2]], "_mr")

# Remove background by applying mask_LM
images_LM[0][mask_LM[0] != 255] = 0
images_LM[1][mask_LM[1] != 255] = 0
images_MR[0][mask_MR[0] != 255] = 0
images_MR[1][mask_MR[1] != 255] = 0

# Display the images
if displayImagesWhileRunning:
    cv.imshow("Left", images_LM[0])
    cv.imshow("Middle", images_LM[1])
    cv.imshow("Middle 2", images_MR[0])
    cv.imshow("Right", images_MR[1])
    cv.waitKey(0)

    # Display the mask
    cv.imshow("Left", mask_LM[0])
    cv.imshow("Middle", mask_LM[1])
    cv.imshow("Middle 2", mask_MR[0])
    cv.imshow("Right", mask_MR[1])
    cv.waitKey(0)

# Generate two meshes
points_LM = generate_point_cloud(images_LM, calibration_data, "lm")
# points_MR = generate_point_cloud(images_MR, calibration_data, "mr")

# point_cloud = np.row_stack([points_LM,points_MR])
# print("pointCloud.shape: ",point_cloud.shape)
# colors = np.ones((point_cloud.shape[0],3),dtype=np.uint8)*255
# create_point_cloud_file(point_cloud,colors, "pointCloud.ply")

# Merge the meshes using the ICP algorithm (iterated closest points)

# Save the mesh to a file
create_mesh(points_LM,"LM",3)
# create_mesh(points_MR,"MR")
# create_mesh(point_cloud,"total")
# Plot the mesh

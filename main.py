"""
Main file that runs the entire pipeline
"""
import numpy as np
import cv2 as cv
from remove_background import get_foreground_mask_HSV
from get_images import getTriplet
from mesh import generate_point_cloud, create_mesh, visualize_point_cloud, visualize_mesh
from normalise_color import normalize_global_color_type
from camera_calibration import get_calibration_data_from_file, rectify_images, calibrate_3_cameras_to_file
from point_cloud_alignment import combine_point_clouds


def preprocess(images, suffix=None):
    mask = np.array([None, None])
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    for i in [0, 1]:
        # Make Foreground mask
        mask[i] = get_foreground_mask_HSV(images[i])

        # global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true.
        images[i] = normalize_global_color_type(images[i])
    # Stereo rectification to facilitate the dense stereo matching, also performs non-linear distortion correction!
    rect_images = rectify_images(images[0], images[1], calibration_data[f'map1x{suffix}'],
                                 calibration_data[f'map1y{suffix}'],
                                 calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'],calibration_data[f'ROI1{suffix}'],calibration_data[f'ROI2{suffix}'], suffix)
    rect_mask = rectify_images(mask[0], mask[1], calibration_data[f'map1x{suffix}'],
                               calibration_data[f'map1y{suffix}'],
                               calibration_data[f'map2x{suffix}'], calibration_data[f'map2y{suffix}'], calibration_data[f'ROI1{suffix}'],calibration_data[f'ROI2{suffix}'], suffix)

    return rect_images, rect_mask#np.array(mask)


# Only execute to generate the calibration data file
#calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

triplet = getTriplet(1, 0)  # In final version we'll loop over these

# Preprocess the images
images_LM, mask_LM = preprocess([triplet[0], triplet[1]], "_lm")
images_MR, mask_MR = preprocess([triplet[1], triplet[2]], "_mr")

# Display the images
# cv.imshow("Left", images_LM[0])
# cv.imshow("Middle", images_LM[1])
# cv.imshow("Middle 2", images_MR[0])
# cv.imshow("Right", images_MR[1])
# cv.waitKey(0)

# Display the mask
# cv.imshow("Left", mask_LM[0])
# cv.imshow("Middle", mask_LM[1])
# cv.imshow("Middle 2", mask_MR[0])
# cv.imshow("Right", mask_MR[1])
# cv.waitKey(0)

#TODO: returns two point clouds, merge them and then create mesh from the combined point cloud

# Generate two point clouds
pc_LM = generate_point_cloud(images_LM, mask_LM, calibration_data, "_lm")
pc_RM = generate_point_cloud(images_MR, mask_MR, calibration_data, "_mr")
print('Finished generating point clouds')
#visualize_point_cloud(pc_LM)

# Merge the point clouds using the ICP algorithm (iterated closest points)
pc_combined = combine_point_clouds(pc_RM, pc_LM)

# Create mesh from the point cloud
mesh = create_mesh(pc_combined, 'pc_combined', 8)
visualize_mesh(mesh)

# Save mesh to file

# Plot the mesh

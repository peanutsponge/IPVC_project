"""
Main file that runs the entire pipeline
"""
from get_images import getTriplet
from point_cloud import generate_point_cloud
from pre_process import preprocess
from camera_calibration import get_calibration_data_from_file, calibrate_3_cameras_to_file
from point_cloud_alignment import combine_point_clouds
from mesh import create_mesh_poisson, visualize_mesh


# Only execute to generate the calibration data file
#calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

# Load the images
triplet = getTriplet(1, 0)  # In final version we'll loop over these

# Preprocess the images
images_LM, mask_LM = preprocess([triplet[0], triplet[1]], calibration_data, "_lm", display=False)
images_MR, mask_MR = preprocess([triplet[1], triplet[2]], calibration_data, "_mr", display=False)
print('Finished preprocessing images')

# Generate two point clouds
pc_LM = generate_point_cloud(images_LM, mask_LM, calibration_data, "_lm", display_disparity=False, display_point_cloud=False)
pc_RM = generate_point_cloud(images_MR, mask_MR, calibration_data, "_mr", display_disparity=False, display_point_cloud=False)
print('Finished generating point clouds')


# Merge the point clouds using the ICP algorithm (iterated closest points)
pc_combined = combine_point_clouds(pc_RM, pc_LM, display=False)

# Create mesh from the point cloud
mesh = create_mesh_poisson(pc_combined, 'pc_combined')
visualize_mesh(mesh)

# Save mesh to file

# Plot the mesh

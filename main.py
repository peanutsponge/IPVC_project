"""
Main file that runs the entire pipeline
"""
from get_images import getTriplet
from point_cloud import generate_point_cloud, visualize_point_cloud
from pre_process import preprocess
from camera_calibration import get_calibration_data_from_file, calibrate_3_cameras_to_file
from point_cloud_alignment import combine_point_clouds
from mesh import create_mesh_poisson, visualize_mesh
from disparity import compute_disparity_map, compute_disparity_map_interactively
import open3d as o3d

# Only execute to generate the calibration data file
# calibrate_3_cameras_to_file('calibration_data.pkl')

# Load the calibration data, see camera_calibration.py for more info on the specific saved dictionary entries
calibration_data = get_calibration_data_from_file('calibration_data.pkl')

subjects = [2]  # Specify the subjects to run the pipeline on
numbers = [0, 1, 2, 3]  # Specify the numbers to run the pipeline on
for s in subjects:
    for n in numbers:
        save_path = f'output/subject{s}/number{n}'
        # Load the images
        triplet = getTriplet(s, n)

        # Preprocess the images
        images_LM, mask_LM = preprocess([triplet[0], triplet[1]], calibration_data, "_lm", save_path, display=False)
        images_MR, mask_MR = preprocess([triplet[1], triplet[2]], calibration_data, "_mr", save_path, display=False)
        print('Finished preprocessing images')

        # Compute the disparity map, note that the first image passed is what the disparity map is based on
        disparity_map_LM = compute_disparity_map(images_LM, "_lm", mask_LM, save_path, display=False)
        disparity_map_MR = compute_disparity_map(images_MR, "_mr", mask_MR, save_path, display=False)
        # compute_disparity_map_interactively(images_LM, mask_LM)
        # compute_disparity_map_interactively(images_MR, mask_MR)
        print('Finished computing disparity maps')

        # # Generate two point clouds
        pc_LM = generate_point_cloud(disparity_map_LM, images_LM, calibration_data, "_lm", display=False)
        pc_MR = generate_point_cloud(disparity_map_MR, images_MR, calibration_data, "_mr", display=False)
        print('Finished generating point clouds')

        # Save the point clouds
        o3d.io.write_point_cloud(f'{save_path}_pc_LM.ply', pc_LM)
        o3d.io.write_point_cloud(f'{save_path}_pc_MR.ply', pc_MR)
        print('Finished saving point clouds')

        # Load the point clouds
        pc_LM = o3d.io.read_point_cloud(f'{save_path}_pc_LM.ply')
        pc_MR = o3d.io.read_point_cloud(f'{save_path}_pc_MR.ply')

        # Merge the point clouds using the ICP algorithm (iterated closest points)
        pc_combined = combine_point_clouds(pc_MR, pc_LM, display=False)
        print('Finished combining point clouds')

        # Save the point cloud
        o3d.io.write_point_cloud(f'{save_path}_pc_combined.ply', pc_combined)
        print('Finished saving point cloud')

        # Load the point cloud
        pc_combined = o3d.io.read_point_cloud(f'{save_path}_pc_combined.ply')

        #  Create mesh from the point cloud
        mesh = create_mesh_poisson(pc_combined, 7, 0.3)
        print('Finished creating mesh')

        # Save mesh to file
        o3d.io.write_triangle_mesh(f'{save_path}_mesh.ply', mesh)
        print('Finished saving mesh')

        # Plot the mesh
        # visualize_mesh(mesh)

"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import open3d as o3d


def visualize_point_cloud(pcd):
    """
    Visualizes a point cloud
    Args:
        pcd:

    Returns:

    """
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Customize visualization settings
    vis.get_render_option().point_size = 1.0  # Adjust point size
    vis.get_render_option().background_color = [1, 1, 1]  # Set background color to black

    # Set the camera viewpoint (optional)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # Adjust the camera's orientation
    view_control.set_up([0, 1, 0])  # Adjust the camera's orientation

    # Capture and render the visualization
    vis.run()
    vis.destroy_window()


def remove_outliers(pcd):
    """
    Removes outliers from a point cloud
    Args:
        pcd:

    Returns:

    """
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.0)
    pcd_outlier_removed = pcd.select_by_index(ind)
    return pcd_outlier_removed


def generate_point_cloud(disparity_map, rectified_images, calibration_data, suffix, display=False):
    """
    Generates a point cloud from a disparity map and the rectified images
    Args:
        disparity_map:
        rectified_images:
        calibration_data:
        suffix:
        display:

    Returns:

    """
    if suffix == "_lm":
        focal_length = (calibration_data['mtx_left'][0][0] + calibration_data['mtx_right'][1][1]) / 2
    elif suffix == "_mr":
        focal_length = (calibration_data['mtx_middle'][0][0] + calibration_data['mtx_middle'][1][1]) / 2

    Q = np.float32([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, focal_length * 0.00325, 0],
                    [0, 0, 0, 1]])

    # Use the disparity map to find the point cloud
    point_cloud = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)
    colors = cv.cvtColor(rectified_images[0], cv.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()  # possibly the same as disparity_map_invalid

    point_cloud = point_cloud[mask_map]
    colors = colors[mask_map]

    # Remove any coordinates that are invalid (i.e. have a value of inf), apply to both the point cloud and the colors
    mask = np.all(np.isfinite(point_cloud), axis=1)
    point_cloud = point_cloud[mask]
    colors = colors[mask]

    # Remove the background plane
    background_points = point_cloud[:, 2] > 0
    point_cloud = point_cloud[background_points]
    colors = colors[background_points]

    # Store the point cloud and colors in a pointcloud object
    pc_obj = o3d.geometry.PointCloud()
    pc_obj.points = o3d.utility.Vector3dVector(point_cloud)
    pc_obj.colors = o3d.utility.Vector3dVector(colors / 255)
    pc_obj = remove_outliers(pc_obj)
    # Visualize the point cloud
    if display:
        visualize_point_cloud(pc_obj)
    return pc_obj

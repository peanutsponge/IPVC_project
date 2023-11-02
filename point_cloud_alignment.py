"""
This file contains functions for aligning and merging point clouds.
"""

import open3d as o3d
import numpy as np
from point_cloud import visualize_point_cloud


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


def merge_point_clouds(pcd1, pcd2, n_x=100, n_y=1000):
    """
    Merges two point clouds by taking the average z value of each grid cell and then taking the point cloud with the
    Args:
        n_y: Resolution in y direction
        n_x: Resolution in x direction
        pcd1: The first point cloud
        pcd2: The second point cloud

    Returns:
        merged_pcd: The merged point cloud
    """
    # Create a grid with size w x h
    x_min = min(pcd1.get_min_bound()[0], pcd2.get_min_bound()[0])
    x_max = max(pcd1.get_max_bound()[0], pcd2.get_max_bound()[0])
    y_min = min(pcd1.get_min_bound()[1], pcd2.get_min_bound()[1])
    y_max = max(pcd1.get_max_bound()[1], pcd2.get_max_bound()[1])

    x_factor = 0.1
    x_range = x_max - x_min
    x_nose = get_nose_x(pcd1)
    x_left = x_nose - x_factor * x_range
    x_right = x_nose + x_factor * x_range

    # Keep everthing left of the nose as pcd1
    pcdleft = pcd2.crop(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=[-np.inf, -np.inf, -np.inf], max_bound=[x_left, np.inf, np.inf]))
    # Keep everthing right of the nose as pcd2
    pcdright = pcd1.crop(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_right, -np.inf, -np.inf], max_bound=[np.inf, np.inf, np.inf]))

    # In the region between x_left and x_right, take the average z value of each grid cell
    x_mid = np.linspace(x_left, x_right, num=n_x)
    y_mid = np.linspace(y_min, y_max, num=n_y)

    merged_pcd = o3d.geometry.PointCloud()

    for i in range(n_x - 1):
        for j in range(n_y - 1):

            x_left, y_left = x_mid[i], y_mid[j]
            x_right, y_right = x_mid[i + 1], y_mid[j + 1]
            # get points in range
            pcd1_frame = pcd1.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_left, y_left, -np.inf],
                                                                       max_bound=[x_right, y_right, np.inf]))
            pcd2_frame = pcd2.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_left, y_left, -np.inf],
                                                                       max_bound=[x_right, y_right, np.inf]))
            if len(pcd1_frame.points) == 0 or len(pcd2_frame.points) == 0:
                continue

            # get average z value
            z = (np.mean(np.asarray(pcd1_frame.points)[:, 2]) * i + np.mean(np.asarray(pcd2_frame.points)[:, 2]) * (
                    n_x - i)) / n_x
            x = (np.mean(np.asarray(pcd1_frame.points)[:, 0]) * i + np.mean(np.asarray(pcd2_frame.points)[:, 0]) * (
                    n_x - i)) / n_x
            y = (np.mean(np.asarray(pcd1_frame.points)[:, 1]) * i + np.mean(np.asarray(pcd2_frame.points)[:, 1]) * (
                    n_x - i)) / n_x
            # get average color
            c = (np.mean(np.asarray(pcd1_frame.colors), axis=0) * i + np.mean(np.asarray(pcd2_frame.colors), axis=0) * (
                    n_x - i)) / n_x

            # create point
            p = np.array([x, y, z])
            p = np.append(p, c)
            p = np.reshape(p, (1, 6))
            pcdframe = o3d.geometry.PointCloud()
            pcdframe.points = o3d.utility.Vector3dVector(p[:, 0:3])
            pcdframe.colors = o3d.utility.Vector3dVector(p[:, 3:6])
            # add to merged point cloud
            merged_pcd += pcdframe
    merged_pcd += (pcdleft + pcdright)
    return merged_pcd


# def merge_point_clouds(pcd1, pcd2, n_x=100, n_y=1000):
#     """
#     Merges two point clouds by taking the average z value of each grid cell and then taking the point cloud with the
#     Args:
#         pcd1:
#         pcd2:
#         n:
#
#     Returns:
#
#     """
#     # Create a grid with size w x h
#     x_min = min(pcd1.get_min_bound()[0], pcd2.get_min_bound()[0])
#     x_max = max(pcd1.get_max_bound()[0], pcd2.get_max_bound()[0])
#     y_min = min(pcd1.get_min_bound()[1], pcd2.get_min_bound()[1])
#     y_max = max(pcd1.get_max_bound()[1], pcd2.get_max_bound()[1])
#
#     x_factor = 0.1
#     x_range = x_max - x_min
#     x_nose = get_nose_x(pcd1)
#     x_left = x_nose - x_factor * x_range
#     x_right = x_nose + x_factor * x_range
#
#     # Keep everthing left of the nose as pcd1
#     pcdleft = pcd2.crop(
#         o3d.geometry.AxisAlignedBoundingBox(min_bound=[-np.inf, -np.inf, -np.inf], max_bound=[x_left, np.inf, np.inf]))
#     # Keep everthing right of the nose as pcd2
#     pcdright = pcd1.crop(
#         o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_right, -np.inf, -np.inf], max_bound=[np.inf, np.inf, np.inf]))
#
#     # In the region between x_left and x_right, take the average z value of each grid cell
#     x_mid = np.linspace(x_left, x_right, num=n_x)
#     y_mid = np.linspace(y_min, y_max, num=n_y)
#
#     merged_pcd = o3d.geometry.PointCloud()
#
#     # Use matrix operations to calculate the average z value of each grid cell
#     z_values = np.zeros((n_x, n_y))
#     weights = np.zeros((n_x, n_y))
#     for pcd in [pcd1, pcd2]:
#         pcd_z = np.asarray(pcd.points)[:, 2]
#         pcd_weights = np.ones(len(pcd_z))
#         for i in range(n_x - 1):
#             for j in range(n_y - 1):
#                 x_left, y_left = x_mid[i], y_mid[j]
#                 x_right, y_right = x_mid[i + 1], y_mid[j + 1]
#                 # Get points in range
#                 pcd_frame = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_left, y_left, -np.inf],
#                                                                         max_bound=[x_right, y_right, np.inf]))
#                 if len(pcd_frame.points) == 0:
#                     continue
#
#                 # Get average z value
#                 z_value = np.mean(pcd_frame.points[:, 2])
#                 # Get weights
#                 weight = len(pcd_frame.points)
#                 # Add to z_values and weights matrices
#                 z_values[i, j] += z_value * weight
#                 weights[i, j] += weight
#
#     # Calculate the average z value of each grid cell
#     z_values /= weights
#
#     # Create a point cloud from the average z values
#     for i in range(n_x):
#         for j in range(n_y):
#             x = x_mid[i]
#             y = y_mid[j]
#             z = z_values[i, j]
#             # Create a point
#             p = np.array([x, y, z])
#             p = p.reshape((1, 3))
#             pcd_frame = o3d.geometry.PointCloud()
#             pcd_frame.points = o3d.utility.Vector3dVector(p)
#             # Add to merged point cloud
#             merged_pcd += pcd_frame
#     merged_pcd += (pcdleft + pcdright)
#     return merged_pcd

def remove_side(pcd, side, factor):
    """
    Removes a side of a point cloud
    Args:
        pcd: the point cloud
        side: left or right
        factor: how much of the side to remove

    Returns:
        pcd: the point cloud with the side removed
    """
    # factor betwen 0-1
    x_min = pcd.get_min_bound()[0]
    x_max = pcd.get_max_bound()[0]
    xrange = x_max - x_min

    if side == 'left':
        # get xrange
        x_min = x_min + factor * xrange
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_min, -np.inf, -np.inf],
                                                           max_bound=[np.inf, np.inf, np.inf]))
    elif side == 'right':
        x_max = x_max - factor * xrange
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[-np.inf, -np.inf, -np.inf],
                                                           max_bound=[x_max, np.inf, np.inf]))
    return pcd


def get_nose_x(pcd, zfactor=0.05):
    """
    Gets the x position of the nose in a point cloud.
    Args:
        pcd: the point cloud
        zfactor: How much of the z range to use to determine the nose

    Returns:

    """
    # find position of nose
    z_max = pcd.get_max_bound()[2]
    z_min = pcd.get_min_bound()[2]
    z_range = z_max - z_min
    # First get only a point cloud with a z value close to the maximum
    pcd_z_max = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[-np.inf, -np.inf, z_max - zfactor * z_range],
                                                             max_bound=[np.inf, np.inf, np.inf]))
    # Locate the x middle of the nose
    x_nose = (pcd_z_max.get_max_bound()[0] + pcd_z_max.get_min_bound()[0]) / 2
    return x_nose


def get_nose_slice(pcd, xfactor, zfactor=0.05):
    """
    Gets the nose slice of a point cloud. Which is the entire vertical slice that includes the nose.
    Args:
        xfactor: The factor of the width of the point cloud that is used to determine the nose slice.
        pcd: The point cloud.
        zfactor: The factor of the width of the point cloud that is used to determine the nose slice.

    Returns:
        pcd_slice: The slice of the point cloud that includes the nose.
    """
    # find position of nose
    x_nose = get_nose_x(pcd, zfactor)
    # Get the width of the point cloud
    x_max = pcd.get_max_bound()[0]
    x_min = pcd.get_min_bound()[0]
    x_range = x_max - x_min
    # Get the width of the nose slice
    x_nose_slice = xfactor * x_range
    # Get the nose slice
    pcd_slice = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_nose - x_nose_slice / 2, -np.inf, -np.inf],
                                                             max_bound=[x_nose + x_nose_slice / 2, np.inf, np.inf]))
    return pcd_slice


def combine_point_clouds(source_pcd_, target_pcd_, display=False):
    """
    Combines two point clouds by aligning them and then merging them.
    Args:
        source_pcd_: The first point cloud
        target_pcd_: The second point cloud
        display: Whether to display the point clouds while aligning and merging them

    Returns:
        merged_pcd: The merged point cloud
    """
    # remove outliers
    source_pcd = remove_outliers(source_pcd_)
    target_pcd = remove_outliers(target_pcd_)

    # copy point clouds, such that the originals are not changed
    src_pcd = get_nose_slice(source_pcd, 0.2)
    tgt_pcd = get_nose_slice(target_pcd, 0.2)

    # Visualize the aligned point clouds
    if display:
        visualize_point_cloud(src_pcd + tgt_pcd)

    # Set the threshold and initial transformation
    threshold = 100  # this is the maximum distance between two correspondences in source and target
    # trans_init = np.identity(4)
    trans_init = np.array([[1, 0.00000000, 0.0, -200],
                           [0.00000000, 1, 0.00000000, 0.00000000],
                           [0, 0.00000000, 1, 0.00000000],
                           [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    # Apply ICP registration to align the source point cloud with the target point cloud
    icp_transformation = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.00001,
            relative_rmse=0.0000001,
            max_iteration=3000  # Replace with the desired number of iterations
        )
    )

    evaluation = o3d.pipelines.registration.evaluate_registration(src_pcd, tgt_pcd, threshold,
                                                                  icp_transformation.transformation)
    print('Mesh evaluation: ', evaluation)

    # Apply the obtained transformation to the source point cloud
    source_pcd.transform(icp_transformation.transformation)

    # Visualize the aligned point clouds
    if display:
        visualize_point_cloud(source_pcd + target_pcd)

    print('Finished aligning point clouds')

    merged_pcd = merge_point_clouds(source_pcd, target_pcd)
    print('Finished merging point clouds')

    if display:
        visualize_point_cloud(merged_pcd)

    return merged_pcd

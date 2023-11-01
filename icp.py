import cv2 as cv
import numpy as np
import open3d as o3d
import copy

from point_cloud import visualize_point_cloud
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def remove_outliers(pcd):
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.0)
    pcd_outlier_removed = pcd.select_by_index(ind)
    return pcd_outlier_removed


def merge_point_clouds(filename_lm, filename_mr):
    """
    Merges two point clouds into one using the iterative closest point algorithm.
    Args:
        point_cloud_lm:
        point_cloud_mr:

    Returns:
        point_cloud: The merged point cloud
    """
    source = o3d.io.read_point_cloud(filename_lm)
    target = o3d.io.read_point_cloud(filename_mr)
    target = remove_outliers(target)
    source = remove_outliers(source)
    # downsample the point clouds
    # source = source.voxel_down_sample(voxel_size=0.001)
    # target = target.voxel_down_sample(voxel_size=0.001)
    # Compute normals for the target point cloud
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # Set the threshold for the correspondence
    threshold = 2000
    trans_init = np.identity(4)
    # draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                                  threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)
    # Apply transformation to the source point cloud
    source.transform(reg_p2p.transformation)
    # Merge the point clouds

    point_cloud = target + source
    # save the merged point cloud
    o3d.io.write_point_cloud("output/merged_point_cloud.ply", point_cloud)

    # plot the merged point cloud
    visualize_point_cloud("output/merged_point_cloud.ply")
    return point_cloud
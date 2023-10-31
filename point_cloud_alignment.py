import open3d as o3d
import numpy as np


def combine_point_clouds(source_pcd_, target_pcd_):
    # remove outliers
    source_pcd = remove_outliers(source_pcd_)
    target_pcd = remove_outliers(target_pcd_)

    # downsample a deep copy of the point cloud
    # source_pcd = source_pcd_.voxel_down_sample(voxel_size=5)
    # target_pcd = target_pcd_.voxel_down_sample(voxel_size=5)

    # Compute normals for the target point cloud
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Set the threshold and initial transformation
    threshold = 0.1  # this is the maximum distance between two correspondences in source and target
    trans_init = np.identity(4)
    #trans_init = np.array([[1, 0.00000000, 0.0, 200],
                           # [0.00000000, 1, 0.00000000, 0.00000000],
                           # [0, 0.00000000, 1, 0.00000000],
                           # [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    # Apply ICP registration to align the source point cloud with the target point cloud
    icp_transformation = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.2,
            relative_rmse=0.00001,
            max_iteration=3000  # Replace with the desired number of iterations
        )
    )

    evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, threshold,
                                                                  icp_transformation.transformation)
    print('Mesh evaluation: ', evaluation)

    print('Transformation matrix: ', icp_transformation.transformation)

    # Apply the obtained transformation to the source point cloud
    source_pcd.transform(icp_transformation.transformation)

    # Visualize the aligned point clouds
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

    # Save the aligned point cloud to a file
    # o3d.io.write_point_cloud("aligned_point_cloud.pcd", source_pcd)

    # Merge the two point clouds
    merged_pcd = source_pcd + target_pcd

    return merged_pcd


def remove_outliers(pcd):
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.0)
    pcd_outlier_removed = pcd.select_by_index(ind)
    return pcd_outlier_removed

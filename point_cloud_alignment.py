import open3d as o3d
import numpy as np
import simpleicp as sicp


def combine_point_clouds(source_pcd_, target_pcd_):
    source_pcd_ = source_pcd_.voxel_down_sample(voxel_size=5)
    target_pcd_ = target_pcd_.voxel_down_sample(voxel_size=5)

    # remove outliers
    source_pcd = remove_outliers(source_pcd_)
    target_pcd = remove_outliers(target_pcd_)

    # downsample a deep copy of the point cloud


    # Compute normals for the target point cloud
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Set the threshold and initial transformation
    threshold = 20000  # this is the maximum distance between two correspondences in source and target
    trans_init = np.identity(4)
    #trans_init = np.array([[1, 0.00000000, 0.0, -200],
    #                      [0.00000000, 1, 0.00000000, 0.00000000],
    #                      [0, 0.00000000, 1, 0.00000000],
    #                      [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    # Apply ICP registration to align the source point cloud with the target point cloud
    icp_transformation = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.00001,
            relative_rmse=0.0000001,
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

def pc_to_pc(pc):

    # Extract the point coordinates from the Open3D point cloud
    open3d_points = pc.points  # Assuming this gives you a Nx3 numpy array

    # Create a SimpleICP PointCloud object and populate it with the extracted points
    simpleicp_pointcloud = sicp.PointCloud(open3d_points, columns=["x", "y", "z"])
    #simpleicp_pointcloud.points = open3d_points
    return simpleicp_pointcloud

def combine_point_clouds2(pc_fix_, pc_mov_):
    pc_fix = pc_to_pc(pc_fix_)
    pc_mov = pc_to_pc(pc_mov_)

    # Create point cloud objects
    #pc_fix = sicp.PointCloud(X_fix, columns=["x", "y", "z"])
    #pc_mov = sicp.PointCloud(X_mov, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    simple_icp = sicp.SimpleICP()
    simple_icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = simple_icp.run(max_overlap_distance=1)

    point_cloud_1 = pc_fix_

    point_cloud_2 = o3d.geometry.PointCloud()
    point_cloud_2.points = o3d.utility.Vector3dVector(X_mov_transformed)
    return point_cloud_1 + point_cloud_2
import open3d as o3d
import numpy as np
from mesh import visualize_point_cloud


def combine_point_clouds(source_pcd_, target_pcd_):
    #TODO: add normalisation and left-right removal of face
    # See code from Damian

    #TODO: first remove outliers, then downsample?
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
    threshold = 2000  # this is the maximum distance between two correspondences in source and target
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
    #o3d.visualization.draw_geometries([source_pcd, target_pcd])

    # Save the aligned point cloud to a file
    # o3d.io.write_point_cloud("aligned_point_cloud.pcd", source_pcd)

    # Merge the two point clouds
    #merged_pcd = source_pcd + target_pcd

    print('Finished combining point clouds')
    merged_pcd = merge_point_clouds(source_pcd, target_pcd)
    visualize_point_cloud(merged_pcd)

    return merged_pcd


def remove_outliers(pcd):
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.0)
    pcd_outlier_removed = pcd.select_by_index(ind)
    return pcd_outlier_removed

def merge_point_clouds(pcd1, pcd2, n=100):
    print('number of points in pcd1: ', len(pcd1.points))
    print('number of points in pcd2: ', len(pcd2.points))
    # Create a grid with size w x h
    x_min = min(pcd1.get_min_bound()[0], pcd2.get_min_bound()[0])
    x_max = max(pcd1.get_max_bound()[0], pcd2.get_max_bound()[0])
    y_min = min(pcd1.get_min_bound()[1], pcd2.get_min_bound()[1])
    y_max = max(pcd1.get_max_bound()[1], pcd2.get_max_bound()[1])

    x = np.linspace(x_min, x_max, num=n)
    y = np.linspace(y_min, y_max, num=n)

    merged_pcd = o3d.geometry.PointCloud()

    for i in range(n-1):
        for j in range(n-1):
            #xrange is between x[i] and x[i+1]
            #yrange is between y[j] and y[j+1]

            #get points in range
            pcd1_frame = pcd1.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x[i], y[j], -np.inf], max_bound=[x[i+1], y[j+1], np.inf]))
            pcd2_frame = pcd2.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[x[i], y[j], -np.inf], max_bound=[x[i+1], y[j+1], np.inf]))

            #get points as numpy array
            pcd1_points = np.asarray(pcd1_frame.points)
            pcd2_points = np.asarray(pcd2_frame.points)

            if len(pcd1_points) == 0 and len(pcd2_points) > 0:
                merged_pcd += pcd2_frame
                continue
            elif len(pcd2_points) == 0 and len(pcd1_points) > 0:
                merged_pcd += pcd1_frame
                continue
            elif len(pcd1_points) == 0 and len(pcd2_points) == 0:
                continue

            # determine which pcd in this range has higher average z value
            pcd1_avg_z = np.mean(pcd1_points[:,2])
            pcd2_avg_z = np.mean(pcd2_points[:,2])

            if pcd1_avg_z > pcd2_avg_z:
                merged_pcd += pcd1_frame
            else:
                merged_pcd += pcd2_frame

    print('number of points in merged pcd: ', len(merged_pcd.points))

    return merged_pcd
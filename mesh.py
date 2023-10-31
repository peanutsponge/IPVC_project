"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import open3d as o3d
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def visualize_mesh(filename):
    point_cloud = o3d.io.read_point_cloud(filename)
    # o3d.visualization.draw_geometries([point_cloud])

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(point_cloud)

    # Customize visualization settings
    vis.get_render_option().point_size = 1.0  # Adjust point size
    vis.get_render_option().background_color = [0, 0, 0]  # Set background color to black

    # Set the camera viewpoint (optional)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # Adjust the camera's orientation
    view_control.set_up([0, 1, 0])  # Adjust the camera's orientation

    # Capture and render the visualization
    vis.run()
    vis.destroy_window()


def create_mesh(points, name, alpha=0.02):
    """
    Create a mesh from a point cloud.
    :param points: The point cloud to create the mesh from
    :param name: The name of the mesh
    :param alpha: The alpha value to use for the mesh
    :return: The mesh
    """
    # if index error, try to change the alpha value
    # Put points into a open3d point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Create a mesh from the point cloud
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    mesh.compute_vertex_normals()
    # plot the mesh
    o3d.visualization.draw_geometries([mesh])
    # Save the mesh to a stl file
    o3d.io.write_triangle_mesh("output/mesh_" + name + ".stl", mesh)
    return mesh


def merge_point_clouds(point_cloud_lm, point_cloud_mr):
    """
    Merges two point clouds into one using the iterative closest point algorithm.
    Args:
        point_cloud_lm:
        point_cloud_mr:

    Returns:
        point_cloud: The merged point cloud
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_cloud_lm)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point_cloud_mr)
    threshold = 0.02
    trans_init = np.asarray([[0.1, 0., 0., 0.],
                             [0., 0.1, 0., 0.],
                             [0., 0., 0.1, 0.],
                             [0., 0., 0., 0.1]])
    draw_registration_result(source, target, trans_init)
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
    return point_cloud

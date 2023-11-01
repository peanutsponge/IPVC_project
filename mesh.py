"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import open3d as o3d
import copy

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

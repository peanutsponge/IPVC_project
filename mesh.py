"""
This file contains functions for creating and visualizing meshes
"""

import open3d as o3d
import numpy as np


def create_mesh_poisson(point_cloud, depth=8, density_threshold=0.02):
    """
    Create a mesh from a point cloud using the poisson algorithm
    Args:
        point_cloud:
        depth: The depth of the octree, the higher the more detail
        density_threshold: The threshold for removing outliers, the higher the more vertices are removed
    Returns:

    """
    # Estimate normals
    point_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Assign some temporary normals
    point_cloud.estimate_normals()
    # o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
    # Create a mesh from the point cloud
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print('Remove mesh outliers (low density)')
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def visualize_mesh(mesh, pointcloud=None):
    """
    Visualize a mesh
    Args:
        mesh:
        pointcloud:

    Returns:

    """
    # plot the mesh
    if pointcloud == None:
        o3d.visualization.draw_geometries([mesh])
    else:
        o3d.visualization.draw_geometries([pointcloud, mesh])

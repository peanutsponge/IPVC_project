"""
This script is used to evaluate the mesh.
"""
from point_cloud_alignment import get_nose_x

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def flatten_mesh_to_xy_plane(mesh):
    """Flattens a mesh to the xy plane.

  Args:
    mesh: An open3d.geometry.TriangleMesh object.

  Returns:
    An open3d.geometry.TriangleMesh object with the mesh flattened to the xy plane.
  """

    # Get the vertices of the mesh.
    vertices = mesh.vertices

    # Project the vertices onto the xy plane.
    projected_vertices = np.copy(vertices)
    projected_vertices[:, 2] = 0
    # turn the projected vertices back into a Vector3dVector
    projected_vertices = o3d.utility.Vector3dVector(projected_vertices)

    # Create a new mesh with the projected vertices.
    new_mesh = o3d.geometry.TriangleMesh(projected_vertices, mesh.triangles)

    return new_mesh


subjects = [2]
numbers = [0, 1, 2, 3]
for s in subjects:
    for n in numbers:
        mesh = o3d.io.read_triangle_mesh(f'output/subject{s}/number{n}_mesh.ply')
        # Evaluate the mesh
        print('Evaluate mesh')
        # convert mesh to point cloud
        pc = mesh.sample_points_uniformly(number_of_points=10000)
        # find the nose tip
        nose_x = get_nose_x(pc)
        # split the mesh into two halves one left and one right of the nose tip
        bounding_box = mesh.get_axis_aligned_bounding_box()
        bounding_box_left = o3d.geometry.AxisAlignedBoundingBox(np.array([nose_x, bounding_box.get_min_bound()[1],
                                                                          bounding_box.get_min_bound()[2]]),
                                                                bounding_box.get_max_bound())
        bounding_box_right = o3d.geometry.AxisAlignedBoundingBox(bounding_box.get_min_bound(),
                                                                 np.array([nose_x, bounding_box.get_max_bound()[1],
                                                                           bounding_box.get_max_bound()[2]]))
        mesh_left = mesh.crop(bounding_box_left)
        mesh_right = mesh.crop(bounding_box_right)
        print("Finished cropping the two meshes")
        # mirror the right mesh along the y-axis such that the two meshes are aligned
        mesh_right = mesh_right.scale(-1, center=(nose_x, 0, 0))
        # Rotate the right mesh 180 degrees around the z-axis such that the two meshes are aligned
        mesh_right = mesh_right.rotate(mesh_right.get_rotation_matrix_from_xyz((0, np.pi, np.pi)),
                                       center=(nose_x, 0, 0))
        print("Finished aligning the two meshes")
        # Create a copy of the left and right mesh flattened to the xy plane
        mesh_left_flat = flatten_mesh_to_xy_plane(mesh_left)
        mesh_right_flat = flatten_mesh_to_xy_plane(mesh_right)

        # compute the z distance between the two meshes for each point in x and y
        # create a grid of points
        x = np.linspace(bounding_box_left.get_min_bound()[0], bounding_box_left.get_max_bound()[0], 100)
        y = np.linspace(bounding_box_left.get_min_bound()[1], bounding_box_left.get_max_bound()[1], 100)
        xx, yy = np.meshgrid(x, y)
        # compute the z distance for each point
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # find the closest point in the left mesh
                pcd_tree_left = o3d.geometry.KDTreeFlann(mesh_left_flat)
                [k, idx, _] = pcd_tree_left.search_knn_vector_3d(np.array([xx[i, j], yy[i, j], 0]), 1)
                closest_point_left = mesh_left.vertices[idx[0]]
                # if the point is too far away in x or y, set the z distance to 200
                distance = np.sqrt((xx[i, j] - closest_point_left[0]) ** 2 + (yy[i, j] - closest_point_left[1]) ** 2)
                if distance > 5:
                    zz[i, j] = np.inf
                    continue

                # find the closest point in the right mesh
                pcd_tree_right = o3d.geometry.KDTreeFlann(mesh_right_flat)
                [k, idx, _] = pcd_tree_right.search_knn_vector_3d(np.array([xx[i, j], yy[i, j], 0]), 1)

                closest_point_right = mesh_right.vertices[idx[0]]
                # if the point is too far away in x or y, set the z distance to 200
                distance = np.sqrt((xx[i, j] - closest_point_right[0]) ** 2 + (yy[i, j] - closest_point_right[1]) ** 2)
                if distance > 5:
                    zz[i, j] = np.inf
                    continue
                # compute the z distance

                zz[i, j] = np.abs(closest_point_left[2] - closest_point_right[2])

        print("Finished computing z distance")
        # compute the average z distance excluding the points with infinite z distance
        zz__ = zz.flatten()
        zz__ = zz__[zz__ != np.inf]

        print(f'Subject {s} number {n} Average z distance: {round(np.mean(zz__), 2)}')
        # plot the z distance as a heatmap
        # invert y axis
        zz_ = np.flipud(zz)
        plt.imshow(zz_)
        plt.colorbar()
        # Let the xaxis tick go from 50 to 100
        plt.xticks(np.linspace(0, 100, 6), np.linspace(50, 100, 6))

        # plt.title(f'Z distance between the two meshes for subject {s} number {n}\nAverage z distance: {round(np.mean(zz__), 2)}')
        plt.xlabel("x (% of mesh width)")
        plt.ylabel("y (% of mesh height)")

        plt.savefig(f'output/subject{s}/number{n}_z_distance.png')
        plt.show()

        # Convert to a point cloud
        zz = zz.flatten()
        xx = xx.flatten()
        yy = yy.flatten()
        points = np.vstack((xx, yy, zz)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # remove the points with infinite z distance
        pcd.remove_non_finite_points()

        # visualize the point cloud and the two meshes in one window
        # o3d.visualization.draw_geometries([pcd, mesh_left, mesh_right])

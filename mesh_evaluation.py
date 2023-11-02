"""
This script is used to evaluate the mesh.
"""

import open3d as o3d

subjects = [2]
numbers = [0]
for s in subjects:
    for n in numbers:
        mesh = o3d.io.read_triangle_mesh(f'output/mesh_subject_{s}_number_{n}.stl')
        # Evaluate the mesh
        print('Evaluate mesh')
        # compute volume of the mesh
        print('Volume: ', mesh.get_volume())
        # compute surface area of the mesh
        print('Surface area: ', mesh.get_surface_area())
        # compute the center of mass of the mesh
        print('Center of mass: ', mesh.get_center_of_mass())
        # compute the inertia tensor of the mesh
        print('Inertia tensor: ', mesh.get_inertia_tensor())
        # compute the vertex normal of the mesh
        print('Vertex normal: ', mesh.get_vertex_normals())
        # compute the triangle normal of the mesh
        print('Triangle normal: ', mesh.get_triangle_normals())
        # compute the triangle area of the mesh
        print('Triangle area: ', mesh.get_triangle_areas())
        # compute the bounding box of the mesh
        print('Bounding box: ', mesh.get_axis_aligned_bounding_box())
        # compute the oriented bounding box of the mesh
        print('Oriented bounding box: ', mesh.get_oriented_bounding_box())


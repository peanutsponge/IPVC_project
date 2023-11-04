import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

angle = 45 / 180 * np.pi
subjects = [1,2,4]
numbers = [0, 1, 2, 3]

def cropImage(path, crop_area):
    '''
    Crop the image (overwrites the original image)
    :param path: The path of the image and where it will be saved
    :param crop_area: The area to be cropped (left, upper, right, lower)
    '''
    # Open the image file
    img = Image.open(path)
    # Crop the image
    img_cropped = img.crop(crop_area)
    # Save the cropped image
    img_cropped.save(path)

for s in subjects:
    for n in numbers:
        mesh = o3d.io.read_triangle_mesh(f'output/subject{s}/number{n}_mesh.ply')

        # Create a window but do not display it
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  
        vis.add_geometry(mesh)

        # Left
        mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, angle, 0)))
        # Visualize the mesh
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        # Save the image
        vis.capture_screen_image(f'output/subject{s}/number{n}_mesh_left.png')
        cropImage(f'output/subject{s}/number{n}_mesh_left.png', (720, 150, 1200, 800))

        # Middle
        mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, -angle, 0)))
        # Visualize the mesh
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        # Save the image
        vis.capture_screen_image(f'output/subject{s}/number{n}_mesh_middle.png')
        cropImage(f'output/subject{s}/number{n}_mesh_middle.png', (720, 150, 1200, 800))
        
        # Right
        mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, -angle, 0)))
        # Visualize the mesh
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        # Save the image
        vis.capture_screen_image(f'output/subject{s}/number{n}_mesh_right.png')
        cropImage(f'output/subject{s}/number{n}_mesh_right.png', (720, 150, 1200, 800))

        vis.destroy_window()
        print(f'Finished saving images for subject {s} number {n}')
print('Finished saving images')


    





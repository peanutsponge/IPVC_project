"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
def generate_mesh(images, mask):
    """
    Use the two images to generate a mesh.
    The images have been stereo rectified, so the epipolar lines are horizontal.
    The images have been compensated for Non-linear lens deformation.
    https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    :param mask: The mask to use to only use the foreground
    :param images: The two images to generate the mesh from
    :return: The mesh
    """
    # Use the two images to find the fundamental matrix

    # Use stereo matching to find the disparity map
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(images[0], images[1])
    # Use the disparity map to find the depth map
    Z = 1 / disparity

    # Use the disparity map to find the depth map

    # Use the depth map to find the 3D points, Skip the points that are not in the mask

    # Use the 3D points to generate a mesh

    # Return the mesh
    return None
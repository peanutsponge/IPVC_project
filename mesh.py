def generate_mesh(im1, im2, mask1, mask2):
    """
    Use the two images to generate a mesh.
    The images have been stereo rectified, so the epipolar lines are horizontal.
    The images have been compensated for Non-linear lens deformation.
    https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    :param mask2:
    :param mask1:
    :param im1:
    :param im2:
    :return: The mesh
    """
    # Use the two images to find the fundamental matrix

    # Use stereo matching to find the disparity map

    # Use the disparity map to find the depth map

    # Use the depth map to find the 3D points, Skip the points that are not in the mask

    # Use the 3D points to generate a mesh

    # Return the mesh
    return None
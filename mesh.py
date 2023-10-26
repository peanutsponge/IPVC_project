"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np


def compute_disparity_map(rectified_images):
    """
    Compute the disparity map from two rectified images.
    :param rectified_images: The two rectified images to compute the disparity map from needs to be in uint8 format
    :return: The disparity map
    """
    block_size = 5
    min_disp = -1
    max_disp = 31
    num_disp = max_disp - min_disp  # Needs to be divisible by 16

    stereo = cv.StereoSGBM_create(
        # Adjust these parameters by trial and error.
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=2,
        disp12MaxDiff=2,
        P1=8 * 3 * block_size ** 2,  # 8*img_channels*block_size**2
        P2=32 * 3 * block_size ** 2  # 32*img_channels*block_size**2
    )

    disparity = stereo.compute(rectified_images[0], rectified_images[1])
    return disparity


def generate_mesh(rectified_images, foreground_masks, calibration_data, suffix):
    """
    Use the two images to generate a mesh.
    The images have been stereo rectified, so the epipolar lines are horizontal.
    The images have been compensated for Non-linear lens deformation.
    https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    :param mask: The mask to use to only use the foreground
    :param images: The two images to generate the mesh from
    :return: The mesh
    """
    # TODO: add foreground mask support
    # TODO: fix point cloud, how to choose points? SIFT?

    gray_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in rectified_images]

    # Compute the disparity map
    disparity_map = compute_disparity_map(rectified_images)  # or use gray_images?

    # Set values smaller than 0 to 0 (these are invalid disparities just like the zeroes)
    disparity_map_invalid = disparity_map <= 0
    disparity_map[disparity_map_invalid] = 0

    # Convert to float32 Why?
    disparity_map = np.float32(np.divide(disparity_map, 16.0))  # Why 16?

    # Show the disparity map
    cv.imshow("Disparity",
              cv.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U))
    cv.waitKey(0)

    # Use the disparity map to find the point cloud
    point_cloud = cv.reprojectImageTo3D(disparity_map, calibration_data[f'Q{suffix}'], handleMissingValues=True)
    colors = cv.cvtColor(rectified_images[0], cv.COLOR_BGR2RGB)  # We don't need colors right?
    mask_map = disparity_map > disparity_map.min()  # possibly the same as disparity_map_invalid

    # Mask the point cloud Why? Don't we want the point cloud as a list of points?
    point_cloud = point_cloud[mask_map]
    colors = colors[mask_map]

    # Create a point cloud file
    create_point_cloud_file(point_cloud, colors, "point_cloud.ply")
    return point_cloud


def create_point_cloud_file(vertices, colors, filename):
    """
    Create a point cloud file from the vertices and colors.
    Args:
        vertices:
        colors:
        filename:

    Returns:

    """
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

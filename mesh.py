"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def compute_disparity_map(rectified_images):
    """
    Compute the disparity map from two rectified images.
    :param rectified_images: The two rectified images to compute the disparity map from needs to be in uint8 format
    :return: The disparity map
    """

    window_size = 30
    min_disp = -1
    max_disp = 31
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=32,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=0,
                                  speckleWindowSize=0,
                                  speckleRange=0
                                  )
    disparity = stereo.compute(rectified_images[0], rectified_images[1])
    return disparity
def compute_disparity_map_graph_cut(rectified_images):
    window_size = 30
    block_size = 5
    min_disp = -1
    max_disp = 47
    num_disp = max_disp - min_disp  # Needs to be divisible by 16

    stereo = cv.StereoSGBM_create(
        # Adjust these parameters by trial and error.
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0,
        disp12MaxDiff=-1,
        P1=8 * 3 * block_size ** 2,  # 8*img_channels*block_size**2
        P2=32 * 3 * block_size ** 2,  # 32*img_channels*block_size**2
        mode=cv.STEREO_SGBM_MODE_HH  # Use Graph Cut mode
    )
    disparity = stereo.compute(rectified_images[0], rectified_images[1])
    return disparity
def compute_disparity_map_block_matching(rectified_images):
    block_size = 5  # Adjust the block size based on your preference
    num_disparities = 16  # Adjust based on the range of disparities in your images
    stereo = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
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



    # Compute the disparity map
    disparity_map = compute_disparity_map_graph_cut(rectified_images)  # or use gray_images?

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

    points = []

    # for i in range(1024):
    #     for j in range(1024):
    #         if disparity_map[i][j] != 0:
    #             point = [j, i, 1/disparity_map[i][j]]
    #             points.append(point)
    # points = np.array(points)
    # # plot the point cloud 2D
    # plt.figure()
    # plt.scatter(points[:, 0], points[:, 1], s=0.1)
    # plt.show()
    # # plot the point cloud
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    return None


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

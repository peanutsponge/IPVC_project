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

def compute_disparity_map_interactively(images, mask):
    """
    Compute the disparity map from two rectified images in an interactive window, where we can adjust the parameters.
    Args:
        images:

    Returns:

    """
    #Default values
    use_mask = False
    colorize = True
    display_range_lower_bound = 0 # The lower bound of the range to display the disparity map
    display_range_upper_bound = 255 # The upper bound of the range to display the disparity map
    block_size = 5
    min_disp = -1
    max_disp = 47
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    uniquenessRatio = 0
    speckleWindowSize = 0
    speckleRange = 0
    disp12MaxDiff = -1
    P1 = 8 * 3 * block_size ** 2  # 8*img_channels*block_size**2
    P2 = 32 * 3 * block_size ** 2  # 32*img_channels*block_size**2
    mode = cv.STEREO_SGBM_MODE_HH  # Use Graph Cut mode

    # Create an interactive window to adjust the parameters
    cv.namedWindow("Disparity", cv.WINDOW_NORMAL)
    cv.createTrackbar("block_size", "Disparity", block_size, 50, lambda x: x)
    cv.createTrackbar("num_disparities", "Disparity", num_disp, 100, lambda x: x * 16)
    cv.createTrackbar("uniquenessRatio", "Disparity", uniquenessRatio, 100, lambda x: x)
    cv.createTrackbar("speckleWindowSize", "Disparity", speckleWindowSize, 200, lambda x: x)
    cv.createTrackbar("speckleRange", "Disparity", speckleRange, 100, lambda x: x)
    cv.createTrackbar("disp12MaxDiff", "Disparity", disp12MaxDiff, 100, lambda x: x)
    cv.createTrackbar("P1", "Disparity", P1, 1000, lambda x: x)
    cv.createTrackbar("P2", "Disparity", P2, 1000, lambda x: x)
    cv.createTrackbar("mode", "Disparity", mode, 1, lambda x: x)
    cv.createTrackbar("display_range_lower_bound", "Disparity", display_range_lower_bound, 1000, lambda x: x)
    cv.createTrackbar("display_range_upper_bound", "Disparity", display_range_upper_bound, 1000, lambda x: x)
    cv.createTrackbar("colorize", "Disparity", colorize,1, lambda x: x)
    cv.createTrackbar("use_mask", "Disparity", use_mask, 1, lambda x: x)

    # Wait until the user presses 'q' on the keyboard
    while cv.waitKey(1) != ord('q'):
        # Get the current trackbar positions
        block_size = cv.getTrackbarPos("block_size", "Disparity")
        num_disp = cv.getTrackbarPos("num_disparities", "Disparity")
        uniquenessRatio = cv.getTrackbarPos("uniquenessRatio", "Disparity")
        speckleWindowSize = cv.getTrackbarPos("speckleWindowSize", "Disparity")
        speckleRange = cv.getTrackbarPos("speckleRange", "Disparity")
        disp12MaxDiff = cv.getTrackbarPos("disp12MaxDiff", "Disparity")
        P1 = cv.getTrackbarPos("P1", "Disparity")
        P2 = cv.getTrackbarPos("P2", "Disparity")
        mode = cv.getTrackbarPos("mode", "Disparity")
        display_range_lower_bound = cv.getTrackbarPos("display_range_lower_bound", "Disparity")
        display_range_upper_bound = cv.getTrackbarPos("display_range_upper_bound", "Disparity")
        colorize = cv.getTrackbarPos("colorize", "Disparity")
        use_mask = cv.getTrackbarPos("use_mask", "Disparity")

        # Create the stereo matcher object with the parameters we set above
        stereo = cv.StereoSGBM_create(
            # Adjust these parameters by trial and error.
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=P1,  # 8*img_channels*block_size**2
            P2=P2,  # 32*img_channels*block_size**2
            mode=mode  # Use Graph Cut mode
        )
        im1 = images[0].copy()
        im2 = images[1].copy()
        if use_mask:
            im1[mask[0] != 255] = [0, 0, 0]
            im2[mask[1] != 255] = [0, 0, 0]

        disparity = stereo.compute(im1, im2)
        # Display the disparity map
        # Convert to float32 Why?
        disparity_map = np.float32(np.divide(disparity, 16.0))  # Why

        if colorize:
            # Color mark everything that is not in the range we want to display
            mask_lower = np.zeros(disparity_map.shape, dtype=np.uint8)
            mask_lower[disparity_map < display_range_lower_bound] = 255
            mask_upper = np.zeros(disparity_map.shape, dtype=np.uint8)
            mask_upper[disparity_map > display_range_upper_bound] = 255
            # Color mark everything that is not in the range we want to display
            disparity_map[disparity_map < display_range_lower_bound] = display_range_lower_bound
            disparity_map[disparity_map > display_range_upper_bound] = display_range_upper_bound
        # Normalize the disparity_map map to the range we want to display
        disparity_map = cv.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                     dtype=cv.CV_8U)
        if colorize:
            # Apply the masks as a color overlay
            disparity_map = cv.applyColorMap(disparity_map, cv.COLORMAP_JET)
            disparity_map[mask_lower == 255] = [0, 0, 0]
            disparity_map[mask_upper == 255] = [255, 255, 255]




        # Show the disparity map in the interactive window let the curser display the disparity value
        cv.imshow("Disparity", disparity_map)
        # Print the disparity value at the current mouse position in the interactive window
        cv.setMouseCallback("Disparity", lambda event, x, y, flags, param: print(disparity[y, x]))
        # wait for 100ms
        cv.waitKey(100)

    # Close the window
    cv.destroyWindow("Disparity")
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
    disparity_map = compute_disparity_map_interactively(rectified_images, mask=foreground_masks)

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

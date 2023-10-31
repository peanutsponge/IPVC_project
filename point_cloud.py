"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import open3d as o3d


def visualize_point_cloud(filename):
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


def create_point_cloud_file(vertices, colors, filename):
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
    # close the file
    f.close()


def compute_disparity_map(images, mask=None):
    """
    Compute the disparity map from two rectified images.
    :param rectified_images: The two rectified images to compute the disparity map from needs to be in uint8 format
    :return: The disparity map
    """
    im1 = images[0].copy()
    im2 = images[1].copy()
    if mask is not None:
        im1[mask[0] != 255] = [0, 0, 0]
        im2[mask[1] != 255] = [0, 0, 0]

    block_size = 5
    num_disp = 48  # Needs to be divisible by 16
    left_matcher = cv.StereoSGBM_create(numDisparities=num_disp,
                                        blockSize=block_size,
                                        P1=8 * 3 * block_size ** 2,
                                        P2=32 * 3 * block_size ** 2,
                                        disp12MaxDiff=100,
                                        uniquenessRatio=5,
                                        speckleWindowSize=9,
                                        speckleRange=2,
                                        mode=cv.STEREO_SGBM_MODE_HH
                                        )

    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    left_disp = left_matcher.compute(im1, im2)
    right_disp = right_matcher.compute(im2, im1)

    wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    #wls_filter.setDepthDiscontinuityRadius(5)
    #wls_filter.setLRCthresh(200)
    disparity = wls_filter.filter(disparity_map_left=left_disp, left_view=im1, disparity_map_right=right_disp)
    disparity = np.float32(np.divide(disparity, 16.0))
    return disparity


def compute_disparity_map_interactively(images, mask):
    # Default values
    use_mask = True
    colorize = True
    display_range_lower_bound = 1  # The lower bound of the range to display the disparity map
    display_range_upper_bound = 255  # The upper bound of the range to display the disparity map
    block_size = 1
    num_disp = 32  # Needs to be divisible by 16
    uniquenessRatio = 2
    speckleWindowSize = 20
    speckleRange = 2
    disp12MaxDiff = 100
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
    cv.createTrackbar("colorize", "Disparity", colorize, 1, lambda x: x)
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
        show_disparity_map(disparity_map, display_range_lower_bound, display_range_upper_bound, colorize)

        # wait for 100ms
        cv.waitKey(100)

    # Close the window
    cv.destroyWindow("Disparity")
    return disparity


def show_disparity_map(disparity_map, display_range_lower_bound, display_range_upper_bound, colorize=True):
    _disparity_map = disparity_map.copy()
    if colorize:
        # Color mark everything that is not in the range we want to display
        mask_lower = np.zeros(_disparity_map.shape, dtype=np.uint8)
        mask_lower[_disparity_map < display_range_lower_bound] = 255
        mask_upper = np.zeros(_disparity_map.shape, dtype=np.uint8)
        mask_upper[_disparity_map > display_range_upper_bound] = 255
        # Color mark everything that is not in the range we want to display
        _disparity_map[_disparity_map < display_range_lower_bound] = display_range_lower_bound
        _disparity_map[_disparity_map > display_range_upper_bound] = display_range_upper_bound
    # Normalize the disparity_map map to the range we want to display
    _disparity_map = cv.normalize(_disparity_map, _disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_8U)
    if colorize:
        # Apply the masks as a color overlay
        _disparity_map = cv.applyColorMap(_disparity_map, cv.COLORMAP_JET)
        _disparity_map[mask_lower == 255] = [0, 0, 0]
        _disparity_map[mask_upper == 255] = [255, 255, 255]

    # Show the disparity map in the interactive window let the curser display the disparity value
    cv.imshow("Disparity", _disparity_map)
    # Print the disparity value at the current mouse position in the interactive window
    cv.setMouseCallback("Disparity", lambda event, x, y, flags, param: print(disparity_map[y, x]))


def normalise_point_cloud(point_cloud):
    """
    Normalise the point cloud to the range [0, 1] centered at the origin.
    Args:
        point_cloud: The point cloud to normalise

    Returns:
        point_cloud: The normalised point cloud
    """
    # makes the mean of the point cloud 0
    point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    # makes the max of the point cloud 1
    point_cloud = point_cloud / np.max(point_cloud)
    return point_cloud


def generate_point_cloud(rectified_images, foreground_masks, calibration_data, suffix):
    # Compute the disparity map, note that the first image passed is what the disparity map is based on
    disparity_map = compute_disparity_map(rectified_images, mask=foreground_masks)

    # Apply first mask to disparity map
    disparity_map[foreground_masks[0] != 255] = 0

    if False:  # If you want to see the disparity map, set this to True
        show_disparity_map(disparity_map, 1, 255)
        cv.waitKey(0)

    # Post processing of the disparity map
    # # A median filter to remove outliers
    # disparity_map = cv.medianBlur(disparity_map, 5)

    if False:  # If you want to see the disparity map, set this to True
        show_disparity_map(disparity_map, 1, 255)
        cv.waitKey(0)

    # important conversion for the reprojectImageTo3D function
    disparity_map = np.float32(np.divide(disparity_map, 16.0))
    print(calibration_data[f'Q{suffix}'])

    if suffix == "_lm":
        focal_length = (calibration_data['mtx_left'][0][0] + calibration_data['mtx_right'][1][1]) / 2
    elif suffix == "_mr":
        focal_length = (calibration_data['mtx_middle'][0][0] + calibration_data['mtx_middle'][1][1]) / 2

    print('focal length: ', focal_length)
    Q = np.float32([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, focal_length * 0.10, 0],
                    [0, 0, 0, 1]])
    # Q = calibration_data[f'Q{suffix}'] # alternative, but sucks

    # Use the disparity map to find the point cloud
    point_cloud = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)
    colors = cv.cvtColor(rectified_images[0], cv.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()  # possibly the same as disparity_map_invalid

    point_cloud = point_cloud[mask_map]
    colors = colors[mask_map]

    # Remove any coordinates that are invalid (i.e. have a value of inf), apply to both the point cloud and the colors
    mask = np.all(np.isfinite(point_cloud), axis=1)
    point_cloud = point_cloud[mask]
    colors = colors[mask]

    # Remove the background plane
    foreground_points = point_cloud[:, 2] > 0
    point_cloud = point_cloud[foreground_points]
    colors = colors[foreground_points]

    # Filter the points that are outside 2 standard deviations
    mean = np.mean(point_cloud, axis=0)
    std = np.std(point_cloud, axis=0) * [1.4, 1.4, 4]  # x y z
    filtered_points = np.all(np.sqrt((point_cloud - mean) ** 2) / std < 1, axis=1)
    point_cloud = point_cloud[filtered_points]
    colors = colors[filtered_points]



    # Filter out outliers
    # filtered_points = filter_regional(point_cloud, n_x=12, n_y=30, threshold=0.5)
    # point_cloud = point_cloud[filtered_points]
    # colors = colors[filtered_points]

    point_cloud = normalise_point_cloud(point_cloud)

    if suffix == "_lm":
        # Filter out the right side of the point cloud
        filtered_points = point_cloud[:, 0] < np.std(point_cloud[:, 0])
    elif suffix == "_mr":
        # Filter out the left side of the point cloud
        filtered_points = point_cloud[:, 0] > -np.std(point_cloud[:, 0])
    point_cloud = point_cloud[filtered_points]
    colors = colors[filtered_points]



    # Create a point cloud file
    create_point_cloud_file(point_cloud, colors, "point_cloud.ply")

    # Visualize the point cloud, comment this out if you don't want to see the point cloud
    visualize_point_cloud("point_cloud.ply")

    return point_cloud


def filter_regional(point_cloud, n_x=9, n_y=15, threshold=1.5):
    # Divide the point cloud into 25 parts (x and y) and filter out the outliers (z) in each part
    # Get the bounds of the point cloud
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    point_cloud_total = np.array([]).reshape(0, 3)
    for i in range(n_x):
        # get bounds of the part
        x_min_region = min_x + (max_x - min_x) / n_x * i
        x_max_region = min_x + (max_x - min_x) / n_x * (i + 1)
        for j in range(n_y):
            # get bounds of the part
            y_min_region = min_y + (max_y - min_y) / n_y * j
            y_max_region = min_y + (max_y - min_y) / n_y * (j + 1)
            # Get points in the part
            regional_point_cloud = point_cloud[(point_cloud[:, 0] > x_min_region) & (point_cloud[:, 0] < x_max_region) &
                                               (point_cloud[:, 1] > y_min_region) & (point_cloud[:, 1] < y_max_region)]
            # Filter out the outliers in the z direction
            mean = np.mean(regional_point_cloud, axis=0)
            std = np.std(regional_point_cloud, axis=0) * [np.inf, np.inf, 1]  # x y z
            filtered_points_region = np.all(np.sqrt((regional_point_cloud - mean) ** 2) / std < threshold, axis=1)
            regional_point_cloud = regional_point_cloud[filtered_points_region]
            # Add the regional_point_cloud points to the total point cloud keeping the order
            point_cloud_total = np.vstack((point_cloud_total, regional_point_cloud))
    # Get indices of the points that are not filtered out in the total point cloud a point is x,y,z
    filtered_points_total = np.all(np.isin(point_cloud, point_cloud_total), axis=1)
    return filtered_points_total

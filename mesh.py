"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from remove_background import region_fill

def visualize_point_cloud_file(filename):
    point_cloud = o3d.io.read_point_cloud(filename)
    #o3d.visualization.draw_geometries([point_cloud])

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

def visualize_point_cloud(pcd):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

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

def create_mesh(point_cloud, name, alpha = 0.02):
    # Create a mesh from the point cloud
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    # Save the mesh to a stl file
    #o3d.io.write_triangle_mesh("output/mesh_"+name+".stl", mesh)
    return mesh

def visualize_mesh(mesh):
    # plot the mesh
    o3d.visualization.draw_geometries([mesh])

def compute_disparity_map_old(images, mask=None):
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
    num_disp = 32
    stereo = cv.StereoSGBM_create(numDisparities=num_disp,
                                  blockSize=block_size,
                                  P1= 8 * 3 * block_size ** 2,
                                  P2= 32 * 3 * block_size ** 2,
                                  disp12MaxDiff=100,
                                  uniquenessRatio=5,
                                  speckleWindowSize=9,
                                  speckleRange=2,
                                  mode = cv.STEREO_SGBM_MODE_HH
                                  )
    disparity = stereo.compute(im1, im2)
    disparity = np.float32(np.divide(disparity, 16.0))
    return disparity

def compute_disparity_map(images, suffix, mask=None):
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

    if suffix == "_lm":
        disparity = wls_filter.filter(disparity_map_left=left_disp, left_view=im2, disparity_map_right=right_disp)
    else:
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

def generate_point_cloud(rectified_images, foreground_masks, calibration_data, suffix):
    # Compute the disparity map, note that the first image passed is what the disparity map is based on
    disparity_map = compute_disparity_map(rectified_images, suffix, mask=foreground_masks)

    # Apply first mask to disparity map
    disparity_map[foreground_masks[0] != 255] = 0

    # If you want to see the disparity map, set this to True
    show_disparity_map = False
    if show_disparity_map:
        _disparity_map = cv.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        _disparity_map = cv.applyColorMap(_disparity_map, cv.COLORMAP_JET)
        cv.imshow("Disparity", _disparity_map)
        cv.waitKey(0)

    # important conversion for the reprojectImageTo3D function
    disparity_map = np.float32(np.divide(disparity_map, 16.0))

    #print(calibration_data[f'Q{suffix}'])

    if suffix == "_lm":
        focal_length = (calibration_data['mtx_left'][0][0] + calibration_data['mtx_right'][1][1]) / 2
    elif suffix == "_mr":
        focal_length = (calibration_data['mtx_middle'][0][0] + calibration_data['mtx_middle'][1][1]) / 2

    Q = np.float32([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, focal_length * 0.05, 0],
                   [0, 0, 0, 1]])
    #Q = calibration_data[f'Q{suffix}'] # alternative, but sucks

    # Use the disparity map to find the point cloud
    point_cloud = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)
    colors = cv.cvtColor(rectified_images[0], cv.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min() # possibly the same as disparity_map_invalid

    point_cloud = point_cloud[mask_map]
    colors = colors[mask_map]

    # Remove any coordinates that are invalid (i.e. have a value of inf), apply to both the point cloud and the colors
    mask = np.all(np.isfinite(point_cloud), axis=1)
    point_cloud = point_cloud[mask]
    colors = colors[mask]

    # Remove the background plane
    background_points = point_cloud[:, 2] > 0
    point_cloud = point_cloud[background_points]
    colors = colors[background_points]

    # Create a point cloud file
    #create_point_cloud_file(point_cloud, colors, "point_cloud.ply")
    #visualize_point_cloud("point_cloud.ply")

    # Store the point cloud and colors in a pointcloud object
    pc_obj = o3d.geometry.PointCloud()
    pc_obj.points = o3d.utility.Vector3dVector(point_cloud)
    pc_obj.colors = o3d.utility.Vector3dVector(colors/255)

    # Visualize the point cloud
    #visualize_point_cloud(pc_obj)

    return pc_obj

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
    #close the file
    f.close()

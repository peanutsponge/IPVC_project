"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(point_cloud):
    # Determine the number of points to sample (10% of the total)
    sample_size = int(0.01 * len(point_cloud))
    # Randomly sample the points
    random_indices = np.random.choice(len(point_cloud), sample_size, replace=False)
    sampled_points = point_cloud[random_indices]
    point_cloud = sampled_points

    x = point_cloud[:, :, 0].flatten()
    y = point_cloud[:, :, 1].flatten()
    z = point_cloud[:, :, 2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')

    plt.show()

def compute_disparity_map(rectified_images):
    #rectified_images = [cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U) for image in rectified_images]
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,  # Adjust as needed for your specific setup
        blockSize=5,  # Adjust as needed for your specific setup
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2
    )

    disparity = stereo.compute(rectified_images[0], rectified_images[1])
    return disparity

def disparity_to_depth(disparity, baseline, focal_length, min_disparity=1e-6):
    # Avoid division by zero and invalid disparities
    mask = (disparity > min_disparity).astype(np.float32)
    depth = baseline * focal_length / (disparity + min_disparity)
    depth *= mask  # Apply the mask to handle invalid disparities
    return depth



def generate_point_cloud(depth_map, calibration_data, camera_name):
    print('Generating point cloud')
    width, height = calibration_data['width'], calibration_data['height']
    f_x = calibration_data['mtx_' + camera_name][0, 0]  # Focal length in X direction
    f_y = calibration_data['mtx_' + camera_name][1, 1]  # Focal length in Y direction
    c_x = calibration_data['mtx_' + camera_name][0, 2]  # Principal point in X direction
    c_y = calibration_data['mtx_' + camera_name][1, 2]  # Principal point in Y direction

    # Create a grid of points
    u, v = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = (u - c_x) * depth_map / f_x
    y = (v - c_y) * depth_map / f_y
    z = depth_map

    # Reshape into a point cloud (3D points)
    point_cloud = np.dstack((x, y, z))

    return point_cloud


def generate_mesh(rectified_images, foreground_masks, calibration_data, camera_name):
    """
    Use the two images to generate a mesh.
    The images have been stereo rectified, so the epipolar lines are horizontal.
    The images have been compensated for Non-linear lens deformation.
    https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    :param mask: The mask to use to only use the foreground
    :param images: The two images to generate the mesh from
    :return: The mesh
    """
    #TODO: add foreground mask support
    #TODO: fix point cloud, how to choose points? SIFT?


    # Compute the disparity map
    disparity_map = compute_disparity_map(rectified_images)
    disparity_map = cv.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imshow("Disparity", disparity_map)
    cv.waitKey(0)

    point_cloud = cv.reprojectImageTo3D(disparity_map, calibration_data['Q_lm'])
    print(point_cloud.shape)
    # Use the disparity map to find the depth map
    #depth_map = disparity_to_depth(disparity_map, 1, 1)
    #mask the depth map
    #depth_map = depth_map * foreground_masks[0]

    # Use the depth map to find the 3D points, Skip the points that are not in the mask
    #point_cloud = generate_point_cloud(depth_map, calibration_data, camera_name)
    plot_point_cloud(point_cloud)
    # Use the 3D points to generate a mesh
    mesh = None

    # Return the mesh
    #for now just return the point cloud
    return point_cloud
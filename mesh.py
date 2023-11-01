"""
This file contains the functions to generate a mesh from two images.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

def create_mesh(points,name, alpha = 2):
    """
    Create a mesh from a point cloud.
    :param points: The point cloud to create the mesh from
    :param name: The name of the mesh
    :param alpha: The alpha value to use for the mesh
    :return: The mesh
    """
    # Put points into a open3d point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Create a mesh from the point cloud
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    mesh.compute_vertex_normals()
    # plot the mesh
    o3d.visualization.draw_geometries([mesh])
    # Save the mesh to a stl file
    o3d.io.write_triangle_mesh("output/mesh_"+name+".stl", mesh)

def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        # Scale the image down by half each time
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

def plot_point_cloud(point_cloud):
    x = point_cloud[:, 0].flatten()
    y = point_cloud[:, 1].flatten()
    z = point_cloud[:, 2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')

    plt.show()

def getPoints(image1,image2,old_points1,old_points2,method):
    if method == "SIFT":
        matchObj = cv.SIFT_create()
        matchMethod = cv.NORM_L1
    elif method == "ORB":
        matchObj = cv.ORB_create()
        matchMethod = cv.NORM_HAMMING
    elif method == "AKAZE":
        matchObj = cv.AKAZE_create(threshold=0.0001)
        matchMethod = cv.NORM_HAMMING
    elif method == _:
        print("Invalid method")
        return

    # Get points
    kp1, des1 = matchObj.detectAndCompute(image1,None)
    kp2, des2 = matchObj.detectAndCompute(image2,None)
    # Match the points
    bf = cv.BFMatcher(matchMethod, crossCheck=True)
    matches = bf.match(des1,des2)
    # Remove matches that do not lay on a epipolar line
    new_matches = []
    for match in matches:
        _, y1 = kp1[match.queryIdx].pt
        _, y2 = kp2[match.trainIdx].pt
        if np.abs(y1 - y2) <= 10:  # In a rectified images the points should be on a horizontal line
            new_matches.append(match)
        old_points1.append(kp1[match.queryIdx].pt)
        old_points2.append(kp2[match.trainIdx].pt)
    matches = new_matches
    # img3 = cv.drawMatches(image1,kp1,image2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    # cv.imshow("Matches:"+method,img3)

def filter_points(points_3d, channel=2, num_stds=2):
        if type(channel) is list:
            for c in channel:
                points_3d = filter_points(points_3d, c, num_stds)
            return points_3d
        z_mean = np.mean(points_3d[:, channel])
        z_std = np.std(points_3d[:, channel])
        z_cutoff = num_stds * z_std
        filtered_points = points_3d[np.abs(points_3d[:, channel]-z_mean) < z_cutoff]
        return filtered_points

def generate_point_cloud(rectified_images, calibration_data, camera_names):
    """
    Use the two images to generate a point cloud.
    The images have been stereo rectified, so the epipolar lines are horizontal.
    The images have been compensated for Non-linear lens deformation.
    https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    :param mask: The mask to use to only use the foreground
    :param images: The two images to generate the point cloud from
    :return: The point cloud
    """
    # Get points from the matches
    points1 = []
    points2 = []
    getPoints(rectified_images[0],rectified_images[1],points1,points2,"ORB")
    getPoints(rectified_images[0],rectified_images[1],points1,points2,"SIFT")
    getPoints(rectified_images[0],rectified_images[1],points1,points2,"AKAZE")
    points1 = np.array(points1)
    points2 = np.array(points2)
    print("Matched points shape: ",points1.shape, points2.shape)  
    # Triangulate the points
    points_4d = cv.triangulatePoints(calibration_data['P1_'+camera_names], calibration_data['P2_'+camera_names], points1.T, points2.T)
    # Convert to 3d points
    points_3d = cv.convertPointsFromHomogeneous(points_4d.T).reshape(-1,3)
    
    # Rotate points to match the camera orientation
    if camera_names == "lm":
        points_3d = np.matmul(points_3d, calibration_data['R_'+camera_names].T)
    else:
        points_3d = np.matmul(points_3d, calibration_data['R_'+camera_names])
    # Filter out outliers in z
    points_3d = filter_points(points_3d,2, 1.5)
    # Rescale z
    points_3d[:,2] /= 4

    print("Points in 3d.shape: ",points_3d.shape)

    # Rotate back
    if camera_names == "lm":
        points_3d = np.matmul(points_3d, calibration_data['R_'+camera_names])
    else:
        points_3d = np.matmul(points_3d, calibration_data['R_'+camera_names].T)

    # Create a point cloud file
    colors = np.ones((points_3d.shape[0],3),dtype=np.uint8)*255
    create_point_cloud_file(points_3d,colors,"point_cloud_"+camera_names+".ply")
    # plot_point_cloud(points_3d)
    # Return the mesh
    #for now just return the point cloud
    return points_3d

def create_point_cloud_file(vertices,colors,filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

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
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')

# Adapted from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
import cv2 as cv
import numpy as np
import os

def readImages(im_array,dir_path):
    filenames = os.listdir(dir_path)
    for filename in filenames:
        im = cv.imread(dir_path+os.path.sep+filename)
        im_array.append(im)

# Load images from the "Calibration 1/calibration.." folder
left_images, middle_images, right_images = [], [], []
readImages(left_images,'Calibratie 1'+os.path.sep+'calibrationLeft')
readImages(middle_images,'Calibratie 1'+os.path.sep+'calibrationMiddle')
readImages(right_images,'Calibratie 1'+os.path.sep+'calibrationRight')

# Detect checkerboard
# parameters
corner_detection_param = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) #criteria
ch_rows = 6  # rows of checkerboard
ch_cols = 9  # columns of checkerboard
square_size = 0.010 # 10mm x 10mm

def calibrate_camera(images, camera_name):
    print('-> Calibrating camera: ', camera_name)
    im_width = images[0].shape[1]
    im_height = images[0].shape[0]

    # Corners in checkerboard space
    corners_ch = np.zeros((ch_rows*ch_cols,3), np.float32)
    corners_ch[:,:2] = np.mgrid[0:ch_cols,0:ch_rows].T.reshape(-1,2) * square_size

    # Corners in image space
    corners_img = []

    # Coordinates of corners in 3D space
    corners_3d = []

    for img in images:
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (ch_cols,ch_rows),None)
        if ret == True:
            corners = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),corner_detection_param) #improves corner accuracy
            cv.drawChessboardCorners(img, (ch_cols,ch_rows), corners,ret)
            #cv.imshow('img', img)
            #k = cv.waitKey(500)
            corners_img.append(corners)
            corners_3d.append(corners_ch)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(corners_3d, corners_img, (im_width,im_height),None,None)
    print('\tCalibration metrics for camera: ', camera_name)
    print('\trmse:', ret)
    #print('camera matrix:\n', mtx)
    #print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
    return mtx, dist, corners_img, corners_3d

mtx_left, dist_left, corners_img_left, corners_3d_left = calibrate_camera(left_images, 'left')
mtx_middle, dist_middle, corners_img_middle, corners_3d_middle = calibrate_camera(middle_images, 'middle')
mtx_right, dist_right, corners_img_right, corners_3d_right = calibrate_camera(right_images, 'right')


#TODO next: translate single camera calibration to stereo calibration, prevent code repeat, see https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
#ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
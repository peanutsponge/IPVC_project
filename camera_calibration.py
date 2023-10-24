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
left_images = []
middle_images = []
right_images = []
readImages(left_images,'Calibratie 1'+os.path.sep+'calibrationLeft')
readImages(middle_images,'Calibratie 1'+os.path.sep+'calibrationMiddle')
readImages(right_images,'Calibratie 1'+os.path.sep+'calibrationRight')
f_h,f_w,_ = left_images[0].shape

#* Detect checkerboard
# parameters
corner_detection_param = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ch_rows = 6  # rows of checkerboard
ch_cols = 9  # columns of checkerboard
square_size = 0.010 # 10mm x 10mm

# Corners in checkerboard space
corners_ch = np.zeros((ch_rows*ch_cols,3), np.float32)
corners_ch[:,:2] = np.mgrid[0:ch_cols,0:ch_rows].T.reshape(-1,2) * square_size

# Corners in image space
corners_img_left = []
corners_img_middle = []
corners_img_right = []

for img in left_images:
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (ch_cols,ch_rows),None)
    if ret == True:
        
        corners_img_left.append(corners)





import cv2 as cv
import numpy as np
import os

def readImages(im_array,dir_path):
    filenames = os.listdir(dir_path)
    for filename in filenames:
        im = cv.imread(dir_path+os.path.sep+filename)
        im_array.append(im)

# Load images from the "Calibration 1/calibrationLeft" folder
# and append them to the "left_images" list
left_images = []
readImages(left_images,'Calibratie 1'+os.path.sep+'calibrationLeft')


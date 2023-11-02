"""
This file contains the code to create a foreground mask for the images.
"""
import os

import cv2 as cv
import numpy as np
from get_images import getTriplet

T = np.uint8  # The type of the images
T_max = np.iinfo(T(10)).max  # The maximum value of the images   

def region_fill(image, seed_point, color):
    """
    Fills a region in an image using floodFill, starting from a seed point
    :param image: image to fill
    :param seed_point: Point to start filling from
    :param color: Color to fill with
    :return: Nothing, overwrites the input image
    """
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    cv.floodFill(image, mask, seed_point, color)


# Settings for when normalized on RGB: s(-2.35) doesn't matter, h_min = 0.6,
def get_foreground_mask_HSV(image, closing_amount = 3, cleaning_amount=9, v_min=0, s_min=127, h_min=40, v_max=255, s_max=255, h_max=200, hue_shift=0, fill_holes=True):
    """
    Returns a binary mask of the foreground of an image based on its HSV values.
    The default parameters are tuned for the NOT normalized images. 
    Because the normalized images are normalized on RGB which caused s channel to be similar to the background
    
    Parameters:
    image (numpy.ndarray<T>): The input image.
    cleaning_amount (T): The number of iterations for the morphological opening operation.
    v_min (T): The minimum value for the V channel in the HSV color space.
    s_min (T): The minimum value for the S channel in the HSV color space.
    h_min (T): The minimum value for the H channel in the HSV color space.
    v_max (T): The maximum value for the V channel in the HSV color space.
    s_max (T): The maximum value for the S channel in the HSV color space.
    h_max (T): The maximum value for the H channel in the HSV color space.

    Returns:
    numpy.ndarray: A binary mask of the foreground of the input image.
    """
    im_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Shift the hue channel
    im_hsv[:, :, 0] = (im_hsv[:, :, 0] + hue_shift) % 256
    # Filter on saturation, value and hue
    mask = cv.inRange(im_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    # Closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=closing_amount)
    # Fill holes
    if fill_holes:
        holes = cv.bitwise_not(mask)
        region_fill(holes, (0, 0), 1)
        region_fill(holes, (round(holes.shape[0]*.7), 0), 1)
        region_fill(holes, (round(holes.shape[0] * .7), round(holes.shape[1] * .7)), 1)
        mask = cv.bitwise_or(mask, holes)
    # Clean up outside
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None, iterations=cleaning_amount)
    return mask

def get_foreground_mask_HSV_interactively(image):
    # Create a window
    cv.namedWindow('Window')
    # Create trackbars with two columns
    cv.createTrackbar('Closing', 'Window', 0, 20, lambda x: x)
    cv.createTrackbar('Cleaning', 'Window', 0, 20, lambda x: x)
    cv.createTrackbar('Fill holes', 'Window', 0, 1, lambda x: x)
    cv.createTrackbar('Hue shift', 'Window', 200, 255, lambda x: x)
    cv.createTrackbar('H min', 'Window', 0, 255, lambda x: x)
    cv.createTrackbar('H max', 'Window', 255, 255, lambda x: x)
    cv.createTrackbar('S min', 'Window', 0, 255, lambda x: x)
    cv.createTrackbar('S max', 'Window', 255, 255, lambda x: x)
    cv.createTrackbar('V min', 'Window', 0, 255, lambda x: x)
    cv.createTrackbar('V max', 'Window', 255, 255, lambda x: x)

    # Loop until the user presses X
    while True:
        # Get the trackbar values
        hue_shift = cv.getTrackbarPos('Hue shift', 'Window')
        h_min = cv.getTrackbarPos('H min', 'Window')
        h_max = cv.getTrackbarPos('H max', 'Window')
        s_min = cv.getTrackbarPos('S min', 'Window')
        s_max = cv.getTrackbarPos('S max', 'Window')
        v_min = cv.getTrackbarPos('V min', 'Window')
        v_max = cv.getTrackbarPos('V max', 'Window')
        cleaning = cv.getTrackbarPos('Cleaning', 'Window')
        closing = cv.getTrackbarPos('Closing', 'Window')
        fill_holes = cv.getTrackbarPos('Fill holes', 'Window')

        im_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # Shift the hue channel
        im_hsv[:, :, 0] = (im_hsv[:, :, 0] + hue_shift) % 256
        # show HSV channels seperately
        im_three = np.concatenate((im_hsv[:, :, 0], im_hsv[:, :, 1], im_hsv[:, :, 2]), axis=1)
        cv.imshow('HSV', im_three)

        # Get the mask
        mask = get_foreground_mask_HSV(image, cleaning_amount=cleaning, closing_amount=closing, v_min=v_min, s_min=s_min, h_min=h_min, v_max=v_max, s_max=s_max, h_max=h_max, hue_shift=hue_shift, fill_holes=fill_holes)

        # Show the mask
        im = image.copy()
        im[mask] = [0, 0, 255]
        im = cv.addWeighted(im, 0.5, image, 0.5, 0)
        cv.imshow('Window', im)

        # wait 100 ms
        cv.waitKey(100)




# Only execute if running the file directly
if __name__ == '__main__':
    # Load images
    images = [getTriplet(i, 0) for i in [1, 2, 4]]
    # Normalize images
    # images = [normalize_global_color_triplet(triplet) for triplet in images]
    print(type(images[0][0][0][0][0]))
    i = 2
    ch = 9
    s_min = 127
    h_min = 40
    h_max = 200
    showMask = True
    # Plot, select the subject with - and = (so + without shift)
    while True:
        if showMask:
            im = get_foreground_mask_HSV(images[i][0],cleaning_amount=ch,s_min=s_min,h_min=h_min,h_max=h_max)
            cv.imshow('Window', im)
        else:
            cv.imshow('Window', images[i][0])

        # Match statement over the key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('-'):
            i = np.max([i - 1, 0])
        elif key == ord('='):
            i = np.min([i + 1, 2])
        elif key == ord('['):
            ch = np.max([ch - 1, 0])
            print(ch)
        elif key == ord(']'):
            ch = np.min([ch + 1, 10]) 
            print(ch)
        elif key == ord('o'):
            s_min = s_min - 10
            print(s_min)
        elif key == ord('p'):
            s_min = s_min + 10
            print(s_min)
        elif key == ord('k'):
            h_min = h_min - 10
            print("h_min",h_min)
        elif key == ord('l'):
            h_min = h_min + 10
            print("h_min",h_min)
        elif key == ord(','):
            h_max = h_max - 10
            print("h_max",h_max)
        elif key == ord('.'):
            h_max = h_max + 10
            print("h_max",h_max)
        elif key == ord('m'):
            showMask = not showMask

    cv.destroyAllWindows()

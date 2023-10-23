"""
This file contains the code to create a foreground mask for the images.
"""
import cv2 as cv
import numpy as np
from get_images import getTriplet
from normalise_color import normalize_global_color_triplet

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
    return image


#! Do not use, Doesn't work with normalized images
def get_foreground_mask(image,threshold = 110, smoothing=2):
    """
    Get the foreground mask for an image. Doesn't work with normalized images
    :param image: image to get the mask for
    :param threshold: threshold value the higher, the less is removed
    :param smoothing: smoothing iterations
    :return: foreground mask
    """
    # Threshold image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(image, threshold, T_max, cv.THRESH_BINARY)
    # Remove some left over background artifacts
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, None, iterations=4)
    mask = cv.bitwise_not(mask)
    # Closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=smoothing)
    # Swap to an inverse mask
    holes = cv.bitwise_not(mask)
    # Isolate the holes (By flooding the background with white)
    region_fill(holes, (image.shape[0] - 1, 0), 1)
    mask = cv.bitwise_or(mask, holes)
    return mask

#! Do not use, Doesn't work with normalized images
def get_foreground_mask_V2(image,channel = 0):
    """
    Get the foreground mask for an image. It uses Otsu thresholding so no threshold value has to be set
    :param image: image to get the mask for
    :return: foreground mask
    """
    # Filter the image with gaussian
    image = cv.GaussianBlur(image, (5, 5), 0)
    # Grayscale using one channel
    image = image[:,:,channel]
    # Otsu thresholding
    _, mask = cv.threshold(image, 0, T_max, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Swap 
    mask = cv.bitwise_not(mask)
    # Closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    # Fill holes
    holes = cv.bitwise_not(mask)
    region_fill(holes, (image.shape[0] - 1, 0), 1)
    mask = cv.bitwise_or(mask, holes)
    # clean up
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, None, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    return mask

# Settings for when normalized on RGB: s(-2.35) doesn't matter, h_min = 0.6,
def get_foreground_mask_HSV(image, cleaning_amount=9, v_min=0, s_min=127, h_min=40, v_max=255, s_max=255, h_max=200):
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
    # Filter on saturation, value and hue
    mask = cv.inRange(im_hsv, (v_min, s_min, h_min), (v_max, s_max, h_max))
    # Closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    # Fill holes
    holes = cv.bitwise_not(mask)
    region_fill(holes, (image.shape[0] - 1, 0), 1)
    mask = cv.bitwise_or(mask, holes)
    # Clean up outside
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None, iterations=cleaning_amount)
    return mask

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

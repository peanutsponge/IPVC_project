import cv2 as cv
import numpy as np
from get_images import getTriplet

def region_fill(image, seed_point, color):
    """Fill a region in an image

    Args:
        image: image to fill
        seed_point (point): Point to start filling from
        color: Color to fill with
    """
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    cv.floodFill(image, mask, seed_point, color)
    return image

def get_foreground_mask(image,smoothing=2):
    # Threshold image
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    _,mask = cv.threshold(image,110,255,cv.THRESH_BINARY)  # 110 is the threshold value the higher the value the less is removed
    # Remove some left over background artifacts
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, None, iterations=4)
    mask = cv.bitwise_not(mask)
    # Closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=smoothing)
    # Swap to an inverse mask
    holes = cv.bitwise_not(mask)
    # Isolate the holes (By flooding the background with white)
    region_fill(holes,(image.shape[0]-1,0),1)
    mask = cv.bitwise_or(mask,holes)
    return mask

# Only execute if running the file directly
if __name__ == '__main__':
    # Load images
    images = [getTriplet(i, 0) for i in [1, 2, 4]]
    i = 0
    # Plot, select the subject with - and = (so + without shift)
    while True:
        im = get_foreground_mask(images[i][0])
        cv.imshow('Left', im)

        # Match statement over the key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('-'):
            i = np.max([i - 1, 0])
        elif key == ord('='):
            i = np.min([i + 1, 2])

    cv.destroyAllWindows()

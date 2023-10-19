import os
import cv2 as cv
import numpy as np

def getTriplet(subject, number):
    '''

    :param subject: 1,2 or 4
    :param number: the number of the file, [0,4]
    :return:
    '''
    images = []
    if subject not in [1,2,4]:
        print("wrong subject number")
        return [None, None, None]
    dir_path_subject = 'subject' + str(subject)
    for perspective in ['Left','Middle','Right']:
        if subject == 1:
            dir_path = f"{dir_path_subject}{os.path.sep}subject{subject}{perspective}"
        else:
            dir_path = f"{dir_path_subject}{os.path.sep}subject{subject}_{perspective}"
        # Change number to the image number
        filenames = os.listdir(dir_path)
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        file_name = filenames[number]
        file_path = f"{dir_path}{os.path.sep}{file_name}"
        if file_name not in os.listdir(dir_path):
            print("Wrong expression number")
            return [None, None, None]
        im = cv.imread(file_path)
        images.append(im)
    return images

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

# Load images
images = [getTriplet(i, 0) for i in [1,2,4]]
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
        i = np.max([i - 1,0])
    elif key == ord('='):
        i = np.min([i + 1,2])
    

cv.destroyAllWindows()
    


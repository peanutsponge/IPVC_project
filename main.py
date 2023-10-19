import os
import cv2 as cv
def getTriplet(subject, number):
    '''

    :param subject: 1,2 or 4
    :param number: the number on the end of the image files
    :return:
    '''
    images = []
    if subject not in [1,2,4]:
        print("wrong subject number")
        return [None, None, None]
    dir_path_subject = 'subject' + str(subject)
    for perspective in ['Left','Middle','Right']:
        dir_path = f"{dir_path_subject}{os.path.sep}subject{subject}{perspective}"
        file_name = f"subject{subject}_{perspective}_{number}.jpg"
        file_path = f"{dir_path}{os.path.sep}{file_name}"
        if file_name not in os.listdir(dir_path):
            print("Wrong expression number")
            return [None, None, None]
        im = cv.imread(file_path)
        images.append(im)
    return images



# Load images from the "Calibration 1/calibrationLeft" folder
# and append them to the "left_images" list
ima = getTriplet(1, 1)

import os
import cv2 as cv


def getTriplet(subject, number):
    """
    Get the triplet of images for a subject and number
    :param subject: 1,2 or 4
    :param number: the number of the file, [0,4]
    :return:
    """
    images = []
    if subject not in [1, 2, 4]:
        print("wrong subject number")
        return [None, None, None]
    if number not in [0, 1, 2, 3]:
        print("wrong expression number")
        return [None, None, None]
    dir_path_subject = 'subject' + str(subject)
    for perspective in ['Left', 'Middle', 'Right']:
        dir_path = f"{dir_path_subject}{os.path.sep}subject{subject}_{perspective}"
        # Change number to the image number
        filenames = os.listdir(dir_path)
        filenames.sort(
            key=lambda x: int(x.split('.')[0].split('_')[-1]))  # Is this needed? Are they not sorted by default
        file_name = filenames[number]
        file_path = f"{dir_path}{os.path.sep}{file_name}"
        if file_name not in os.listdir(dir_path):
            print("Wrong expression number")
            return [None, None, None]
        im = cv.imread(file_path)
        images.append(im)
    return images

import cv2 as cv
import numpy as np
from google.protobuf.json_format import MessageToDict  # Used to convert protobuf message to a dictionary.


def make_mask_for_image(frame, lower, upper):
    """
    mask of image in HSV dimension
    :param frame: image
    :param lower: lower threshold [, , , ]
    :param upper: uppper threshold [, , , ]
    :return: masked image
    """
    lower = np.array(lower)
    upper = np.array(upper)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask_range = cv.inRange(hsv_frame, lower, upper)

    return mask_range


def get_detected_hand(results):
    """
    -1: No hands are detected
    0: both hands are detected
    1: right hand ist detected
    2: left hand ist detected
    :return: int
    """
    number_of_hand = -1
    if len(results.multi_handedness) == 2:
        number_of_hand = 0
    else:
        for i in results.multi_handedness:
            label = MessageToDict(i)['classification'][0]['label']
            if label == 'Right':
                number_of_hand = 1
            if label == 'Left':
                number_of_hand = 2

    return number_of_hand

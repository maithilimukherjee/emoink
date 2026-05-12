import cv2
import numpy as np


def calculate_slant_angle(image):
    """
    Calculate average handwriting slant angle.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        float: Average slant angle in degrees.
    """

    # convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find contours
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    angles = []

    for contour in contours:

        # ignore tiny noise
        if cv2.contourArea(contour) < 20:
            continue

        # fit minimum area rectangle
        rect = cv2.minAreaRect(contour)

        angle = rect[-1]

        # normalize angle
        if angle < -45:
            angle = 90 + angle

        angles.append(angle)

    # avoid empty contour list
    if len(angles) == 0:
        return 0.0

    # average slant angle
    average_angle = np.mean(angles)

    return round(average_angle, 2)
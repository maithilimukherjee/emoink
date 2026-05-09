# 1. Convert the image to grayscale

import cv2

def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Grayscale image.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image
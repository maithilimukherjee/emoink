# 3. Adaptive thresholding for the denoised grayscale handwriting image

import cv2

def adaptive_thresholding(image):
    """
    Apply adaptive Gaussian thresholding to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Binary image after thresholding.
    """

    binary_image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return binary_image
import cv2
import numpy as np


def segment_lines(image):
    """
    Segment handwritten text lines from a binary image.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        list: List of segmented line images.
    """

    # horizontal projection
    horizontal_projection = np.sum(image < 255, axis=1)

    lines = []
    start = None

    for i, value in enumerate(horizontal_projection):

        # detect text region
        if value > 0 and start is None:
            start = i

        # detect end of text region
        elif value == 0 and start is not None:

            line = image[start:i, :]

            lines.append(line)

            start = None

    return lines


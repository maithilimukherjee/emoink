import cv2
import numpy as np


def calculate_baseline_deviation(image):
    """
    Calculate handwriting baseline deviation.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        float: Baseline deviation score.
    """

    # grayscale conversion if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape

    baseline_points = []

    # scan each column
    for x in range(width):

        column = image[:, x]

        # handwriting pixels in column
        ys = np.where(column < 255)[0]

        if len(ys) > 0:

            # bottom-most handwriting pixel
            baseline_y = np.max(ys)

            baseline_points.append(baseline_y)

    # insufficient baseline points
    if len(baseline_points) < 2:
        return 0.0

    # deviation of baseline
    deviation = np.std(baseline_points)

    return round(deviation, 2)
import cv2
import numpy as np


def calculate_stroke_smoothness(image):
    """
    Calculate handwriting stroke smoothness.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        float: Stroke smoothness score.
    """

    # grayscale conversion if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find contours
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    smoothness_scores = []

    for contour in contours:

        # ignore tiny contours
        if cv2.contourArea(contour) < 20:
            continue

        # contour perimeter
        perimeter = cv2.arcLength(contour, True)

        # polygon approximation
        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # avoid division by zero
        if len(approx) == 0:
            continue

        # smoothness ratio
        smoothness = perimeter / (
    len(approx) * np.sqrt(cv2.contourArea(contour))
)

        smoothness_scores.append(smoothness)

    # insufficient contours
    if len(smoothness_scores) == 0:
        return 0.0

    # average smoothness
    average_smoothness = np.mean(smoothness_scores)

    return round(average_smoothness, 2)
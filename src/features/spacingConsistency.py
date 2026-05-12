import cv2
import numpy as np


def calculate_spacing_consistency(image):

    """
    Calculate spacing consistency of handwriting.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        float: Spacing consistency score (0 to 1).
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

    # sort contours left to right
    contours = sorted(
        contours,
        key=lambda c: cv2.boundingRect(c)[0]
    )

    centers = []

    for contour in contours:

        # ignore tiny noise
        if cv2.contourArea(contour) < 20:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        centers.append((x + w / 2, y + h / 2))

    # insufficient contours
    if len(centers) < 2:
        return 1.0

    distances = []

    for i in range(len(centers) - 1):

        dist = np.linalg.norm(
            np.array(centers[i]) -
            np.array(centers[i + 1])
        )

        distances.append(dist)

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if mean_dist == 0:
        return 1.0

    consistency_score = max(
        0,
        min(1, 1 - (std_dist / mean_dist))
    )

    return round(consistency_score, 4)
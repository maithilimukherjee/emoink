import cv2
import numpy as np

def deskew_image(image):
    """
    Deskew a binary handwriting image.

    Parameters:
        image (numpy.ndarray): Binary image.

    Returns:
        numpy.ndarray: Deskewed image.
    """

    coords = np.column_stack(np.where(image < 255))

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    rotation_angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        rotation_angle,
        1.0
    )

    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed
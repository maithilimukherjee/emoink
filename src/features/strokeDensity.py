import numpy as np
import cv2
def calculate_stroke_density(image):
    """
    Calculate the stroke density of a handwritten image.

    Parameters:
    image (numpy.ndarray): The input image (grayscale or binary).

    Returns:
    float: The stroke density (number of black pixels per unit area).
    """
    # Ensure the image is binary (0 and 255)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the number of black pixels (assuming black is 0)
    black_pixels = np.sum(image == 0)

    # Calculate the total area of the image
    total_area = image.shape[0] * image.shape[1]

    # Avoid division by zero
    if total_area == 0:
        return 0.0

    # Calculate stroke density
    stroke_density = black_pixels / total_area

    return round(stroke_density,4)

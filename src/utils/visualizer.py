import cv2
from matplotlib import pyplot as plt


def show_image(image, title="Image", cmap='gray'):
    """
    Display a single image.

    Parameters:
        image (numpy.ndarray): Image to display.
        title (str): Window title.
        cmap (str): Color map.
    """

    plt.figure(figsize=(8, 6))

    plt.imshow(image, cmap=cmap)

    plt.title(title)

    plt.axis('off')

    plt.show()
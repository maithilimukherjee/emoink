import cv2

def denoise_image(image):
    """
    Apply Non-local Means Denoising to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Denoised grayscale image.
    """

    denoised_image = cv2.fastNlMeansDenoising(
        image,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return denoised_image
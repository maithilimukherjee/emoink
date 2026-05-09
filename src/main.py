# load image, grayscale, denoise, threshold, deskew if required,
# save outputs to output folder

import cv2
import numpy as np
import os

from utils.imageloader import load_image

from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.denoise import denoise_image
from src.preprocessing.threshold import adaptive_thresholding
from src.preprocessing.deskew import deskew_image


def main():

    # create output directories if they don't exist
    os.makedirs('outputs/grayscale', exist_ok=True)
    os.makedirs('outputs/denoised', exist_ok=True)
    os.makedirs('outputs/thresholded', exist_ok=True)

    # load image
    image = load_image('input/image.jpg')

    # grayscale conversion
    gray = convert_to_grayscale(image)
    cv2.imwrite('outputs/grayscale/grayscale_image.jpg', gray)

    # denoise image
    denoised = denoise_image(gray)
    cv2.imwrite('outputs/denoised/denoised_image.jpg', denoised)

    # adaptive thresholding
    thresh = adaptive_thresholding(denoised)
    cv2.imwrite('outputs/thresholded/threshold_image.jpg', thresh)

    # detect skew angle
    coords = np.column_stack(np.where(thresh < 255))

    # deskew if sufficient skew detected
    if len(coords) > 0:

        angle = cv2.minAreaRect(coords)[-1]

        if abs(angle) > 5:
            processed = deskew_image(thresh)
        else:
            processed = thresh

    else:
        processed = thresh

    # save final processed image
    cv2.imwrite('outputs/processed_image.jpg', processed)


if __name__ == "__main__":
    main()
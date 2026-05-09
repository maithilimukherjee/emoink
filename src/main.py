# load image, preprocess handwriting image,
# segment text lines, visualize outputs,
# save outputs to output folder

import cv2
import numpy as np
import os

from utils.imageloader import load_image
from utils.visualizer import visualize_pipeline

from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.denoise import denoise_image
from src.preprocessing.threshold import adaptive_thresholding
from src.preprocessing.deskew import deskew_image
from src.preprocessing.segment import segment_lines


def main():

    # create output directories
    os.makedirs('outputs/grayscale', exist_ok=True)
    os.makedirs('outputs/denoised', exist_ok=True)
    os.makedirs('outputs/thresholded', exist_ok=True)
    os.makedirs('outputs/lines', exist_ok=True)

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

    # deskew if needed
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

    # segment handwriting lines
    lines = segment_lines(processed)

    # save segmented lines
    for idx, line in enumerate(lines):

        cv2.imwrite(
            f'outputs/lines/line_{idx + 1}.jpg',
            line
        )

    # visualize preprocessing pipeline
    visualize_pipeline([
        ("Original", image),
        ("Grayscale", gray),
        ("Denoised", denoised),
        ("Thresholded", thresh),
        ("Processed", processed)
    ])

    print(f"Total segmented lines: {len(lines)}")


if __name__ == "__main__":
    main()
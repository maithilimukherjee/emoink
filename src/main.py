import cv2
import numpy as np
import os

from utils.imageloader import load_image
from utils.visualizer import show_image

from preprocessing.grayscale import convert_to_grayscale
from preprocessing.denoise import denoise_image
from preprocessing.threshold import adaptive_thresholding
from preprocessing.deskew import deskew_image
from preprocessing.segment import segment_lines
from features.extractfeats import extract_features
from inference.normalizescores import normalize_scores
from inference.affectivestate import infer_affective_state

INPUT_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"


def process_image(image_path, save_prefix):

    image = load_image(image_path)

    # grayscale
    gray = convert_to_grayscale(image)

    # denoise
    denoised = denoise_image(gray)

    # threshold
    thresh = adaptive_thresholding(denoised)

    # deskew
    coords = np.column_stack(np.where(thresh < 255))

    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]

        if abs(angle) > 5:
            processed = deskew_image(thresh)
        else:
            processed = thresh
    else:
        processed = thresh

    # save processed image
    cv2.imwrite(f"{OUTPUT_DIR}/{save_prefix}_processed.jpg", processed)

    # segmentation
    lines = segment_lines(processed)

    line_dir = f"{OUTPUT_DIR}/{save_prefix}_lines"
    os.makedirs(line_dir, exist_ok=True)

    for i, line in enumerate(lines):
        cv2.imwrite(f"{line_dir}/line_{i+1}.jpg", line)

    return image, gray, denoised, thresh, processed, len(lines)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"found {len(files)} images")

    for idx, file in enumerate(files):

        image_path = os.path.join(INPUT_DIR, file)

        print(f"\nprocessing: {file}")

        image, gray, denoised, thresh, processed, n_lines = process_image(
            image_path,
            save_prefix=f"img_{idx}"
        )

        # visualize only first image (avoid spam)
        if idx == 0:
            show_image(image, title="Original")
            show_image(gray, title="Grayscale")
            show_image(denoised, title="Denoised")
            show_image(thresh, title="Thresholded")
            show_image(processed, title="Processed")
        
        extracted_features = extract_features(processed)
        print(f"extracted features: {extracted_features}")
        
        affective_scores = infer_affective_state(extracted_features)
        
        normalized_scores = normalize_scores(affective_scores)
        print(f"normalized probabilities: {normalized_scores}")
        


if __name__ == "__main__":
    main()
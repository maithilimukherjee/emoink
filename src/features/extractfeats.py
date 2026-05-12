from features.strokeDensity import calculate_stroke_density

from features.slantAngle import calculate_slant_angle

from features.spacingConsistency import (
    calculate_spacing_consistency
)

from features.baselineDev import (
    calculate_baseline_deviation
)

from features.strokeSmoothness import (
    calculate_stroke_smoothness
)


def extract_features(image):
    """
    Extract handwriting features from image.

    Parameters:
        image (numpy.ndarray): Binary handwriting image.

    Returns:
        dict: Extracted feature vector.
    """

    features = {

        "stroke_density":
            calculate_stroke_density(image),

        "slant_angle":
            calculate_slant_angle(image),

        "spacing_consistency":
            calculate_spacing_consistency(image),

        "baseline_deviation":
            calculate_baseline_deviation(image),

        "stroke_smoothness":
            calculate_stroke_smoothness(image)
    }

    return features
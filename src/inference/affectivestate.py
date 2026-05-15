def infer_affective_state(features):
    """
    Infer affective handwriting state
    from extracted handwriting features.

    Parameters:
        features (dict): Handwriting feature vector.

    Returns:
        dict: Affective state scores.
    """

    scores = {
        "calm": 0,
        "tense": 0,
        "lazy": 0,
        "angry": 0,
        "artistic": 0
    }

    density = float(features["stroke_density"])
    slant = abs(float(features["slant_angle"]))
    spacing = float(features["spacing_consistency"])
    smoothness = float(features["stroke_smoothness"])

    # -------------------------
    # CALM
    # -------------------------

    if 0.08 <= density <= 0.18:
        scores["calm"] += 2

    if spacing > 0.4:
        scores["calm"] += 2

    if smoothness > 1.8:
        scores["calm"] += 1

    if slant < 10:
        scores["calm"] += 1

    # -------------------------
    # TENSE
    # -------------------------

    if density > 0.16:
        scores["tense"] += 2

    if spacing < 0.25:
        scores["tense"] += 2

    if slant > 15:
        scores["tense"] += 1

    if smoothness < 1.5:
        scores["tense"] += 1

    # -------------------------
    # LAZY / LOW EFFORT
    # -------------------------

    if density < 0.08:
        scores["lazy"] += 2

    if spacing > 0.7:
        scores["lazy"] += 1

    if smoothness < 1.2:
        scores["lazy"] += 1

    # -------------------------
    # ANGRY
    # -------------------------

    if density > 0.17:
        scores["angry"] += 2

    if smoothness < 1.4:
        scores["angry"] += 2

    if slant > 18:
        scores["angry"] += 1

    # -------------------------
    # ARTISTIC
    # -------------------------

    if 10 <= slant <= 20:
        scores["artistic"] += 2

    if 0.3 <= spacing <= 0.6:
        scores["artistic"] += 2

    if smoothness > 2:
        scores["artistic"] += 1

    return scores
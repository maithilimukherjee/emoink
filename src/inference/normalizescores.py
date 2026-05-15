def normalize_scores(scores):
    """
    Normalize affective scores into probabilities.

    Parameters:
        scores (dict): Raw affective scores.

    Returns:
        dict: Normalized probabilities.
    """

    total = sum(scores.values())

    # avoid division by zero
    if total == 0:

        n = len(scores)

        return {
            emotion: round(1 / n, 4)
            for emotion in scores
        }

    probabilities = {

        emotion: round(score / total, 4)

        for emotion, score in scores.items()
    }

    return probabilities
import numpy as np

EMOTIONS = ["calm", "tense", "lazy", "angry", "artistic"]

# random-ish but structured weights (temporary baseline)
W = np.array([
    [-0.8, -0.2, -0.3, -0.1],  # calm
    [ 0.6,  0.4,  0.2,  0.5],  # tense
    [-0.5, -0.6, -0.4, -0.3],  # lazy
    [ 0.7,  0.8,  0.3,  0.6],  # angry
    [ 0.3, -0.1,  0.6,  0.7],  # artistic
])

b = np.array([0.2, 0.1, 0.05, 0.15, 0.2])


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def infer_affective_state(features):
    x = np.array([
        features["stroke_density"],
        features["slant_angle"],
        features["spacing_consistency"],
        features["stroke_smoothness"]
    ])

    logits = W @ x + b
    probs = softmax(logits)

    return {
        EMOTIONS[i]: float(probs[i])
        for i in range(len(EMOTIONS))
    }
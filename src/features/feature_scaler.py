import numpy as np

class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std
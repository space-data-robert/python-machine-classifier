import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class TransformLabel(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()

        output = np.where(output == 1, 0, output)
        output = np.where(output == -1, 1, output)
        return output

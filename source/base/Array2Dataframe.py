import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class Array2Dataframe(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)

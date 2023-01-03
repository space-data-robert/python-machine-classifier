from sklearn.base import TransformerMixin, BaseEstimator


class SelectFeature(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, target_name=None):
        self.feature_names = feature_names
        self.target_name = target_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.target_name is None:
            return X[self.feature_names]
        return X[self.feature_names + [self.target_name]]

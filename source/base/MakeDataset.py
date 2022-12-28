from sklearn.base import TransformerMixin, BaseEstimator


class MakeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, train_len):
        self.train_len = train_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        train = X.iloc[:self.train_len, :]
        valid = X.iloc[self.train_len:, :]
        return train, valid

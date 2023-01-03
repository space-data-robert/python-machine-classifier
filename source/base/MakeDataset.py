from sklearn.base import TransformerMixin, BaseEstimator


class MakeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, train_len, target_name):
        self.train_len = train_len
        self.target_name = target_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x_train = X.iloc[:self.train_len, :].copy()
        y_train = x_train.pop(
            self.target_name
        )
        return x_train, y_train
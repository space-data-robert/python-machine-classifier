import pacmap
from sklearn.base import TransformerMixin, BaseEstimator


class EmbedPacmap(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.cluster = pacmap.PaCMAP(
            n_components=len(self.feature_names),
            n_neighbors=None,
            MN_ratio=0.5,
            FP_ratio=2.0,
            num_iters=10**3,
            verbose=False,
            random_state=27,
            save_tree=True,
        )

    def fit(self, X, y=None):
        self.cluster.fit(
            X[self.feature_names],
            init='pca'
        )
        return self

    def transform(self, X):
        output = self.cluster.transform(
            X[self.feature_names]
        )
        return output

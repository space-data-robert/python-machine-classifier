import gc
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore')

from autoimpute.imputations import SingleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline

from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline as make_pipeline_imb

from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

from source.base import *
from source.anomaly import *


def pipeline_clean(feature_names: list(), target_name: str, train_len: int):
    pipeline = make_pipeline(
        SingleImputer(
            strategy={'Class': 'binary logistic'},
            predictors={'Class': 'all'},
            seed=27
        ),
        SelectFeature(
            feature_names=feature_names,
            target_name=[target_name]
        ),
        MakeDataset(
            train_len=train_len
        )
    )
    return pipeline

def pipeline_predict(feature_names, contamination):
    iforest = make_pipeline(
        EmbedPacmap(
            feature_names=feature_names,
            num_iters=50,
        ),
        CustomIsolationForest(
            n_estimators=1000,
            contamination=contamination)
    )

    kneighbor = make_pipeline_imb(
        SelectFeature(feature_names=feature_names),
        PowerTransformer(),
        RobustScaler(),
        SMOTETomek(random_state=42),
        KNeighborsClassifier(
            n_neighbors=6,
            weights='distance',
            metric='manhattan',
            algorithm='brute',
            leaf_size=33,
            p=2)
    )

    lgbm = make_pipeline_imb(
        SelectFeature(feature_names=feature_names),
        PowerTransformer(),
        RobustScaler(),
        SMOTETomek(random_state=42),
        lgb.LGBMClassifier(
            learning_rate=0.05,
            n_estimators=1000,
            random_state=27)
    )

    pipeline = VotingClassifier([
        ('IsolationForest', iforest),
        ('KNeighborsClassifier', kneighbor),
        ('LGBMClassifier', lgbm)],
        voting='soft',
        weights=[1, 3, 2]
    )
    return pipeline


if __name__ == '__main__':

    feature_names = list([
        'V3', 'V7', 'V14', 'V17', 'V16'
    ])
    target_name = 'Class'

    df_train = pd.read_csv(
        'data/train.csv'
    )
    df_train_len = len(df_train)
    print(f'>>> train length = {df_train_len}')

    df_valid = pd.read_csv(
        'data/valid.csv'
    )

    df_data = pd.concat(
        [df_train, df_valid],
        axis=0
    )

    cleaner = pipeline_clean(
        feature_names, target_name, df_train_len
    )
    df_train_x, df_train_y = cleaner.fit_transform(df_data)

    df_valid_x = df_valid.copy()
    df_valid_y = df_valid_x.pop(
        target_name
    )
    del df_train, df_valid
    gc.collect()

    normal_cnt, fraud_cnt = df_valid_y.value_counts()

    contamination = (fraud_cnt / normal_cnt) * 1.2
    print(f'>>> contamination = {contamination: .5f}')

    predictor = pipeline_predict(
        feature_names, contamination
    )
    predictor.fit(df_train_x, df_train_y)

    df_pred_y = predictor.predict(df_valid_x)

    num_score = f1_score(
        df_valid_y,
        df_pred_y,
        average='macro'
    )
    print(f'>>>f1 score = {num_score: .5f}')

    print(classification_report(df_valid_y, df_pred_y))

    print(confusion_matrix(df_valid_y, df_pred_y))

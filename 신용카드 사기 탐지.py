import pandas as pd
from autoimpute.imputations import SingleImputer
from sklearn.pipeline import make_pipeline


def pipeline_clean(feature_names, target_name):
    return make_pipeline(
        # TransoformBoxCox(),
        # StandardScaler(),
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


if __name__ == '__main__':
    FEATURE_NAMES = ['V3', 'V7', 'V14', 'V17', 'V16']
    TARGET_NAME = 'Class'

    df_train = pd.read_csv('data/train.csv')
    df_valid = pd.read_csv('data/valid.csv')

    df_data = pd.concat(
        [df_train, df_valid],
        axis=0
    )

    pipeline_clean = pipeline_clean(
        FEATURE_NAMES,
        TARGET_NAME
    )

    train, valid = pipeline_clean.fit_transform(df_data)

""" Before installed category_encoders & catboost. """

import warnings
# import random
import numpy as np
import pandas as pd
# from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
# from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
# from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10)


def pipeline():
    data_nm = 'data/credit.csv'
    # 데이터를 불러옵니다.
    data: pd.DataFrame = pd.read_csv(data_nm)

    print(f'data.shape = {data.shape}')
    data.head(2)

    # 이상치를 대체합니다.
    data.fillna('NaN', inplace=True)

    # 가족 구성 인원이 큰 경우를 제거합니다.
    data = data.loc[data.family_size <= 7]

    # 모두 같은 값을 지녔으므로 제거합니다.
    data.drop('flag_mobil', axis=1, inplace=True)

    # 취업을 하지 않은 경우 0 으로 대체합니다.
    data.days_employed = data.days_employed.apply(
        lambda x: 0 if x >= 0 else x
    )

    # 현재까지 출생일, 업무 시작일, 신용카드 발급 월을 양수로 변경합니다.
    for column_nm in ['days_birth', 'begin_month', 'days_employed']:
        data[column_nm] = data[column_nm].apply(
            lambda x: abs(x)
        )

    # 취업 준비 기간을 나타냅니다.
    data['days_setup_employed'] = data.days_birth - data.days_employed

    # 취업 준비 기간에 따른 수입 비율입니다.
    data['income_total_per_days_setup_employed'] = data.income_total / data.days_setup_employed

    # 취업 월과 주차를 표기합니다.
    data['employed_month'] = np.floor(data.days_setup_employed / 30) % 12
    # 모두 같은 기준을 적용했으므로 카테고리화해서 사용할 수 있습니다.
    data['employed_week'] = np.floor(data.days_setup_employed / 7) % 4

    # 나이 파생 변수를 생성합니다.
    data['age'] = data.days_birth // 365
    # 태어난 월과 주차를 만듭니다.
    data['age_month'] = np.floor(data.days_birth / 30) % 12
    data['age_week'] = np.floor(data.days_birth / 7) % 4

    # 한사람의 능력 변수를 만듭니다.
    data['income_total_per_days_birth_and_employed'] = (
            data.income_total / (data.days_birth + data.days_employed)
    )

    # 부양 가족수 대비 수입을 나타냅니다.
    data['income_total_per_family_size'] = data.income_total / data.family_size

    # 아이디 변수에 필요한 컬럼을 추출합니다.
    user_id_for_info_nm: list = [
        'child_num', 'income_total', 'days_birth', 'days_employed',
        'work_phone', 'phone', 'email', 'family_size', 'gender', 'car', 'reality',
        'income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type'
    ]
    # 한사람이 여러개 카드를 만들수 있기에 begin_month 제외합니다.
    data['user_id'] = data[user_id_for_info_nm].astype(str).apply(
        lambda x: '_'.join(x), axis=1
    )

    # 다중공선성 컬럼을 추출하고 제거합니다.
    multicollinear_column_nm: list = ['child_num', 'days_birth', 'days_employed', ]
    data.drop(
        multicollinear_column_nm,
        axis=1,
        inplace=True)

    # 피처와 타겟 데이터를 분할합니다.
    x_data: pd.DataFrame = data.copy()
    # 타겟 이름을 나타냅니다.
    target_nm = 'credit'
    # 타겟 변수를 만듭니다.
    y_data = x_data.pop(target_nm)

    # 연속형 변수 컬럼을 추출합니다.
    numerical_column_nm: list = list(
        x_data.dtypes[x_data.dtypes != 'object'].index)
    # 이어서 범주형 변수 컬럼입니다.
    categorical_column_nm: list = list(
        x_data.dtypes[x_data.dtypes == 'object'].index)

    # 로그 스케일링을 합니다.
    x_data.income_total = np.log1p(x_data.income_total)

    # 범주형 변수를 인코딩합니다.
    ordinal_encoder = OrdinalEncoder(categorical_column_nm)
    # 타겟에 따라 딕셔너리 형태의 인코딩입니다.
    x_data[categorical_column_nm] = ordinal_encoder.fit(
        x_data[categorical_column_nm],
        y_data)
    # 피팅 모델로 인코딩을 진행합니다.
    x_data[categorical_column_nm] = ordinal_encoder.transform(
        x_data[categorical_column_nm]
    )
    # 아이디는 정수 형태로 변환합니다.
    x_data.user_id = x_data.user_id.astype('int64')

    # 클러스터로 데이터를 그루핑합니다.
    kmeans = KMeans(
        n_clusters=36,
        random_state=42)
    # 클러스터 모델을 학습합니다.
    kmeans.fit(x_data)
    # 클러스터 그룹을 예측합니다.
    x_data['cluster_id'] = kmeans.predict(x_data)

    # 로그 스케일 바탕의 소득을 제외한 나머지 변수를 정규화합니다.
    numerical_column_nm.remove('income_total')
    # 정규화를 진행합니다.
    standard_scaler = StandardScaler()
    x_data[numerical_column_nm] = standard_scaler.fit_transform(
        x_data[numerical_column_nm]
    )
    # TODO: catboost modeling & pipelining
    return 0


if __name__ == '__main__':
    pipeline()






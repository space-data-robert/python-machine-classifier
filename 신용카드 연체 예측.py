import numpy as np
import pandas as pd


def pipeline():
    # 데이터를 로드합니다.
    data: pd.DataFrame = pd.read_csv('data/credit.csv')
    print(f'data.shape = {data.shape}')
    # 이상치를 대체합니다.
    data.fillna('NaN', inplace=True)
    # 가족 구성 인원이 큰 경우를 제거합니다.
    data = data.loc[data.family_size < 8]
    # 모두 같은 값을 지녔으므로 제거합니다.
    data.drop('flag_mobil', axis=1, inplace=True)
    # 취업을 하지 않은 경우 0으로 대체합니다.
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
    data['month_employed'] = np.floor(data.days_setup_employed / 30) % 12
    data['week_employed'] = np.floor(data.days_setup_employed / 7) % 4

    return 0


if __name__ == '__main__':
    pipeline()






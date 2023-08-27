import pandas as pd


def preprocess(df):
    df_X = df.drop(columns='Attrition', errors='ignore')

    # 使わないデータの削除
    unique_drop_list = ['id']
    drop_list = ['Over18', 'StandardHours', 'EmployeeNumber']
    df_X = df_X.drop(columns=unique_drop_list)
    df_X = df_X.drop(columns=drop_list)

    category_list = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'Gender',
        'JobRole',
        'MaritalStatus',
        'OverTime'
    ]
    category_dict = dict()
    for cate in category_list:
        categories = df_X[cate].unique().tolist()
        df_X[cate] = pd.Categorical(df_X[cate], categories=categories)
        category_dict[cate] = categories

    df_X_categories = pd.get_dummies(df_X[category_list], drop_first=True)

    df_X_others = df_X.select_dtypes(exclude='category')
    df_X_others['SumSatisfaction'] = df_X_others[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].sum(axis=1)
    df_X_others = df_X_others.drop(columns=['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction'])

    features = [
        'DistanceFromHome',
        'SumSatisfaction',
        'HourlyRate',
        'PercentSalaryHike',
        'Age',
        'DailyRate',
        'MonthlyIncome',
        'YearsWithCurrManager',
        'JobLevel',
        'NumCompaniesWorked',
        'YearsAtCompany',
        'YearsInCurrentRole',
        'OverTime_Yes',
        # 'TotalWorkingYears',
        'StockOptionLevel'
    ]
    df_X = pd.concat([df_X_others, df_X_categories], axis=1)[features]

    return df_X
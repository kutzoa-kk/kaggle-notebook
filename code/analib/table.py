import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoost, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics


class XGBoost:
    def __init__(self,
                 X_train, y_train,
                 X_valid, y_valid,
                 X_test, y_test):
        # 学習用
        self.train_data = xgb.DMatrix(X_train, label=y_train)
        # 検証用
        self.valid_data = xgb.DMatrix(X_valid, label=y_valid)
        # テスト用
        self.test_data = xgb.DMatrix(X_test, label=y_test)

    def train(self, params=None):
        # パラメータを設定
        if params is None:
            self.params = {
                'objective': 'multi:softprob',  # 多値分類問題
                'num_class': 4,                 # 目的変数のクラス数
                'learning_rate': 0.01,           # 学習率
                'eval_metric': 'mlogloss'       # 学習用の指標 (Multiclass logloss)
            }
        else:
            self.params = params

        # 学習
        evals = [(self.train_data, 'train'), (self.valid_data, 'eval')]  # 学習に用いる検証用データ
        evaluation_results = {}                            # 学習の経過を保存する箱
        bst = xgb.train(self.params,                       # 上記で設定したパラメータ
                        self.train_data,                    # 使用するデータセット
                        num_boost_round=20000,             # 学習の回数
                        early_stopping_rounds=100,         # アーリーストッピング
                        evals=evals,                       # 学習経過で表示する名称
                        evals_result=evaluation_results,   # 上記で設定した検証用データ
                        verbose_eval=0                     # 学習の経過の表示(非表示)
                        )

        # テストデータで予測
        y_pred = bst.predict(self.test_data)
#         y_pred_max = np.argmax(y_pred, axis=1)

#         accuracy = accuracy_score(self.test_data.get_label(), y_pred_max)
#         print('XGBoost Accuracy:', accuracy)

        return (bst, y_pred)


class LightGBM:
    def __init__(self,
                 X_train, y_train,
                 X_valid=None, y_valid=None,
                 X_test=None, y_test=None):
        # 学習用
        self.train_data = lgb.Dataset(X_train, y_train,
                                      free_raw_data=False)
        # 検証用
        self.valid_data = lgb.Dataset(X_valid, y_valid,
                                      reference=self.train_data,
                                      free_raw_data=False)
        self.test_data = lgb.Dataset(X_test, y_test, free_raw_data=False).construct()

    def train(self, params=None):
        # パラメータを設定
        if params is None:
            self.params = {
                'task': 'train',                # レーニング ⇔　予測predict
                'boosting_type': 'gbdt',        # 勾配ブースティング
                'objective': 'multiclass',      # 目的関数：多値分類、マルチクラス分類
                'metric': 'multi_logloss',      # 検証用データセットで、分類モデルの性能を測る指標
                'num_class': 4,                 # 目的変数のクラス数
                'learning_rate': 0.01,           # 学習率（初期値0.1）
                'num_leaves': 23,               # 決定木の複雑度を調整（初期値31）
                'min_data_in_leaf': 1,          # データの最小数（初期値20）
                'verbosity': -1
            }
        else:
            self.params = params

        # 学習
        evaluation_results = {}                                # 学習の経過を保存する箱
        model = lgb.train(
            self.params,                              # 上記で設定したパラメータ
            self.train_data,                      # 使用するデータセット
            num_boost_round=20000,               # 学習の回数
            valid_names=['train', 'valid'],      # 学習経過で表示する名称
            valid_sets=[self.train_data, self.valid_data],    # モデルの検証に使用するデータセット
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(0),            # この数字を1にすると学習時のスコア推移がコマンドライン表示される
                lgb.record_evaluation(evaluation_results),
            ])

        # テストデータで予測
        y_pred = model.predict(self.test_data.get_data(), num_iteration=model.best_iteration)
        # if self.params['objective'] == 'multiclass':
        #     y_pred_max = np.argmax(y_pred, axis=1)

        # Accuracy の計算
        # accuracy = sum(self.test_data.get_label() == y_pred_max) / len(self.test_data.get_label())
        # print('LightGBM Accuracy:', accuracy)

        return (model, y_pred)
        # return(model, y_pred_max, accuracy)

    def cv(self, params=None, n_splits=5):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_result = lgb.cv(
            params,
            self.train_data,
            num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
            folds=cv,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=10,
                    verbose=True), # early_stopping用コールバック関数
                lgb.log_evaluation(0)] # コマンドライン出力用コールバック関数
        )
        return cv_result


class CatBoostModel:
    def __init__(self,
                 X_train, y_train,
                 X_valid, y_valid,
                 X_test, y_test):
        # 学習用
        self.train_data = Pool(X_train, label=y_train)
        # 検証用
        self.valid_data = Pool(X_valid, label=y_valid)
        self.test_data = Pool(X_test, y_test)

    def train(self, params=None):
        # パラメータを設定
        if params is None:
            self.params = {
                'loss_function': 'MultiClass',    # 多値分類問題
                'num_boost_round': 20000,          # 学習の回数
                'early_stopping_rounds': 100       # アーリーストッピングの回数
            }
        else:
            self.params = params

        # 学習
        model = CatBoost(self.params)
        model.fit(self.train_data, eval_set=[self.valid_data], verbose=False)

        # テストデータで予測
        # y_pred = catb.predict(self.test_data, prediction_type='Class')

        return model

    def turning(self, X_train, y_train):
        skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=42)
        model = CatBoostClassifier()
        # パラメーターを設定する
        param_grid = {'depth': [4, 7, 10],
                      'learning_rate' : [0.01, 0.1],
                      'l2_leaf_reg': [4, 9],
                      'iterations': [500, 1000, 10000]}
        # パラメータチューニングをグリッドサーチで行うために設定する
        ## このGridSearchCV には注意が必要 scoring は そのスコアを基準にして最適化する
        grid_result = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=skf,
                                   verbose=0,
                                   return_train_score=False,
                                   n_jobs=-1)
        grid_result.fit(self.train_data.get_features(), self.train_data.get_label())
        return grid_result


def benchmark(X, y):
    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    # 学習データを学習用と検証用に分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=y_train)
    
    # RandomForest
    # rf_params = {
    #     'random_state': 42
    #     'n_estimators': 10,
    # }
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    rf_y_pred = rf_model.predict(X_test)
    rf_acc = metrics.accuracy_score(y_test, rf_y_pred)
    print(f'RandomForest Accuracy score: {rf_acc}')
    
    # XGBoost
    xgb_params = {
        'random_state': 42,
        'objective': 'binary:logistic',  # 多値分類問題
        # 'num_class': 4,                 # 目的変数のクラス数
        # 'learning_rate': 0.01,           # 学習率
        # 'max_depth': 6,
        # 'eval_metric': 'mlogloss',       # 学習用の指標 (Multiclass logloss)
        'verbosity': 0
    }
    xgbc = XGBoost(X_train, y_train,
                   X_valid, y_valid,
                   X_test, y_test)
    xgb_model, xgb_y_prob = xgbc.train(params=xgb_params)
    
    xgb_y_pred = (xgb_y_prob > 0.5).astype(int)
    xgb_acc = metrics.accuracy_score(y_test, xgb_y_pred)
    print(f'XGBoost Accuracy score: {xgb_acc}')
    
    # LightGBMの学習を実行
    lgbm_params = {
        'random_state': 42,
        'task': 'train',                # レーニング ⇔　予測predict
        'boosting_type': 'gbdt',        # 勾配ブースティング
        'objective': 'binary',          # 目的関数：多値分類binaly、マルチクラス分類mulit_class
        'metric': 'binary_logloss',     # 検証用データセットで、分類モデルの性能を測る指標
        # 'num_class': 4,               # 目的変数のクラス数
        # 'learning_rate': 0.01,           # 学習率（初期値0.1）
        # 'n_estimators': 10000,           # 木の数　default 100
        # 'num_leaves': 31,               # 決定木の複雑度を調整（初期値31）
        # 'max_depth': -1,                # default -1
        # 'min_data_in_leaf': 1,          # データの最小数（初期値20）
        'verbosity': -1
    }
    lgbm = LightGBM(X_train, y_train,
                    X_valid, y_valid,
                    X_test, y_test)
    lgb_model, lgb_y_prob = lgbm.train(params=lgbm_params)

    lgb_y_pred = (lgb_y_prob > 0.5).astype(int)
    lgb_acc = metrics.accuracy_score(y_test, lgb_y_pred)
    print(f'LightGBM Accuracy score: {lgb_acc}')
    
    # CatBoost
    catb_params = {
        'random_state': 42,
        'loss_function': 'Logloss',    # 多値分類問題
        # 'num_boost_round': 20000,          # 学習の回数
        # 'early_stopping_rounds': 100       # アーリーストッピングの回数
    }
    catc = CatBoostModel(X_train, y_train,
                         X_valid, y_valid,
                         X_test, y_test)
    catb_model = catc.train(params=catb_params)
    catb_y_prob = catb_model.predict(X_test, prediction_type='Class')
    catb_y_pred = (catb_y_prob > 0.5).astype(int)
    catb_acc = metrics.accuracy_score(y_test, catb_y_pred)
    print(f'CatBoost Accuracy score: {catb_acc}')

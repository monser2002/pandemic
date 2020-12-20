import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


def rmsle(actual, prediction):
    return np.sqrt(mean_squared_log_error(actual + 1, prediction + 1))


def tune_model(x_train, y_train):
    model = XGBRegressor(tree_method='gpu_hist')
    pipe = Pipeline([
        ('model', model)
    ])
    param_grid = {
        'model__max_depth': range(5, 25, 1),
        'model__reg_alpha': [1.1, 1.2, 1.3],
        'model__reg_lambda': [1.1, 1.2, 1.3],
        'model__n_estimators': range(100, 500, 100),
        'model__objective': ['reg:squaredlogerror'],
        'model__colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'model__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
        'model__min_child_weight': range(5, 10),
        'model__subsample': [0.5, 0.6, 0.7]}

    rs_keras = RandomizedSearchCV(pipe,
                                  param_grid,
                                  scoring=make_scorer(rmsle, greater_is_better=False),
                                  refit=True,
                                  verbose=3,
                                  # n_jobs=-1,
                                  random_state=21,
                                  cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=21))
    rs_keras.fit(x_train, y_train)
    print('{}: \n{}\n{}: \n{}'.format('best_params_', rs_keras.best_params_, 'best_score_', rs_keras.best_score_))


if __name__ in '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.warnings.filterwarnings('ignore')
    tf.get_logger().setLevel('ERROR')
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    os.environ['PYTHONHASHSEED'] = '0'
    seed = 21
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)

    lbl = LabelEncoder()
    train_a = pd.read_csv('train_region_group_a.csv')
    train_a['region_group'] = lbl.fit_transform(train_a['region_group'].astype(str))
    train = pd.read_csv('train_features.csv', index_col=0).drop(['institution'], axis=1)
    train_a_merged = pd.merge(train_a, train, on='id')
    test_a = pd.read_csv('test_region_group_a.csv')
    test_a['region_group'] = lbl.fit_transform(test_a['region_group'].astype(str))
    test = pd.read_csv('test_features.csv', index_col=0).drop(['institution'], axis=1)
    test_a_merged = pd.merge(test_a, test, on='id')

    X_train_a = train_a_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1)
    y_train_a = train_a_merged.salary_x
    X_test_a = test_a_merged.drop(['salary_desired_y'], axis=1)
    # X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    #     train_a_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1), train_a_merged.salary_x,
    #     test_size=0.25, random_state=seed)

    # tune_model(X_train_a, y_train_a)
    model_xg_a = XGBRegressor(subsample=0.7, reg_lambda=1.1, reg_alpha=1.3, objective='reg:squaredlogerror',
                              max_depth=20, n_estimators=200, min_child_weight=4, learning_rate=0.05,
                              colsample_bytree=0.8)
    model_xg_a.fit(X_train_a, y_train_a)
    model_xg_a_second = XGBRegressor(subsample=0.6, reg_lambda=1.3, reg_alpha=1.3, objective='reg:squaredlogerror',
                                     max_depth=20, n_estimators=200, min_child_weight=5, learning_rate=0.04,
                                     colsample_bytree=0.8)
    model_xg_a_second.fit(X_train_a, y_train_a)

    my_scorer = make_scorer(rmsle, greater_is_better=False)
    model_gbm_a = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
    model_gbm_a.loss_ = my_scorer
    model_gbm_a.fit(X_train_a, y_train_a)

    estimators_a = [('XGBoost', model_xg_a),
                    ('GBM', model_gbm_a),
                    ('XGBoostSecond', model_xg_a_second)]
    regressor_a = StackingRegressor(estimators=estimators_a, final_estimator=GradientBoostingRegressor())
    regressor_a.fit(X_train_a, y_train_a)
    predictions_a = pd.DataFrame(regressor_a.predict(X_test_a), index=X_test_a.index, columns=['salary'])

    train_b = pd.read_csv('train_region_group_b.csv')
    train_b['region_group'] = lbl.fit_transform(train_b['region_group'].astype(str))
    train_b_merged = pd.merge(train_b, train, on='id')
    test_b = pd.read_csv('test_region_group_b.csv')
    test_b['region_group'] = lbl.fit_transform(test_b['region_group'].astype(str))
    test_b_merged = pd.merge(test_b, test, on='id')

    X_train_b = train_b_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1)
    y_train_b = train_b_merged.salary_x
    X_test_b = test_b_merged.drop(['salary_desired_y'], axis=1).fillna(0)
    # X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    #     train_b_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1), train_b_merged.salary_x,
    #     test_size=0.25, random_state=seed)

    model_xg_b = XGBRegressor(subsample=0.7, reg_lambda=1.1, reg_alpha=1.3, objective='reg:squaredlogerror',
                              max_depth=20, n_estimators=200, min_child_weight=4, learning_rate=0.05,
                              colsample_bytree=0.8)
    model_xg_b.fit(X_train_b, y_train_b)
    model_xg_b_second = XGBRegressor(subsample=0.6, reg_lambda=1.3, reg_alpha=1.3, objective='reg:squaredlogerror',
                                     max_depth=20, n_estimators=200, min_child_weight=5, learning_rate=0.04,
                                     colsample_bytree=0.8)
    model_xg_b_second.fit(X_train_a, y_train_a)
    model_gbm_b = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
    model_gbm_b.loss_ = my_scorer
    model_gbm_b.fit(X_train_b, y_train_b)

    estimators_b = [('XGBoost', model_xg_b),
                    ('GBM', model_gbm_b),
                    ('XGBoostSecond', model_xg_b_second)]
    regressor_b = StackingRegressor(estimators=estimators_b, final_estimator=GradientBoostingRegressor())
    regressor_b.fit(X_train_b, y_train_b)
    predictions_b = pd.DataFrame(regressor_b.predict(X_test_b), index=X_test_b.index, columns=['salary'])

    train_c = pd.read_csv('train_region_group_c.csv')
    train_c['region_group'] = lbl.fit_transform(train_c['region_group'].astype(str))
    train_c_merged = pd.merge(train_c, train, on='id')
    test_c = pd.read_csv('test_region_group_c.csv')
    test_c['region_group'] = lbl.fit_transform(test_c['region_group'].astype(str))
    test_c_merged = pd.merge(test_c, test, on='id')

    X_train_c = train_c_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1)
    y_train_c = train_c_merged.salary_x
    X_test_c = test_c_merged.drop(['salary_desired_y'], axis=1).fillna(0)
    # X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    #     train_c_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1), train_c_merged.salary_x,
    #     test_size=0.25, random_state=seed)
    model_xg_c = XGBRegressor(subsample=0.7, reg_lambda=1.1, reg_alpha=1.3, objective='reg:squaredlogerror',
                              max_depth=20, n_estimators=200, min_child_weight=4, learning_rate=0.05,
                              colsample_bytree=0.8)
    model_xg_c.fit(X_train_c, y_train_c)
    model_xg_c_second = XGBRegressor(subsample=0.6, reg_lambda=1.3, reg_alpha=1.3, objective='reg:squaredlogerror',
                                     max_depth=20, n_estimators=200, min_child_weight=5, learning_rate=0.04,
                                     colsample_bytree=0.8)
    model_xg_c_second.fit(X_train_a, y_train_a)
    model_gbm_c = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
    model_gbm_c.loss_ = my_scorer
    model_gbm_c.fit(X_train_c, y_train_c)

    estimators_c = [('XGBoost', model_xg_c),
                    ('GBM', model_gbm_c),
                    ('XGBoostSecond', model_xg_c_second)]
    regressor_c = StackingRegressor(estimators=estimators_c, final_estimator=GradientBoostingRegressor())
    regressor_c.fit(X_train_c, y_train_c)
    predictions_c = pd.DataFrame(regressor_c.predict(X_test_c), index=X_test_c.index, columns=['salary'])

    train_d = pd.read_csv('train_region_group_d.csv')
    train_d['region_group'] = lbl.fit_transform(train_d['region_group'].astype(str))
    train_d_merged = pd.merge(train_d, train, on='id')
    test_d = pd.read_csv('test_region_group_d.csv')
    test_d['region_group'] = lbl.fit_transform(test_d['region_group'].astype(str))
    test_d_merged = pd.merge(test_d, test, on='id')

    X_train_d = train_d_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1)
    y_train_d = train_d_merged.salary_x
    X_test_d = test_d_merged.drop(['salary_desired_y'], axis=1).fillna(0)
    # X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    #     train_d_merged.drop(['salary_x', 'salary_y', 'salary_desired_y'], axis=1), train_d_merged.salary_x,
    #     test_size=0.25, random_state=seed)
    model_xg_d = XGBRegressor(subsample=0.7, reg_lambda=1.1, reg_alpha=1.3, objective='reg:squaredlogerror',
                              max_depth=20, n_estimators=200, min_child_weight=4, learning_rate=0.05,
                              colsample_bytree=0.8)
    model_xg_d.fit(X_train_d, y_train_d)
    model_xg_d_second = XGBRegressor(subsample=0.6, reg_lambda=1.3, reg_alpha=1.3, objective='reg:squaredlogerror',
                                     max_depth=20, n_estimators=200, min_child_weight=5, learning_rate=0.04,
                                     colsample_bytree=0.8)
    model_xg_d_second.fit(X_train_a, y_train_a)
    model_gbm_d = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
    model_gbm_d.loss_ = my_scorer
    model_gbm_d.fit(X_train_d, y_train_d)
    estimators_d = [('XGBoost', model_xg_d),
                    ('GBM', model_gbm_d),
                    ('XGBoostSecond', model_xg_d_second)]
    regressor_d = StackingRegressor(estimators=estimators_d, final_estimator=GradientBoostingRegressor())
    regressor_d.fit(X_train_d, y_train_d)
    predictions_d = pd.DataFrame(regressor_d.predict(X_test_d), index=X_test_d.index, columns=['salary'])

    preds = pd.concat([predictions_a, predictions_b, predictions_c, predictions_d])

    preds.salary = preds.salary.apply(lambda x: 12500 if x < 1000 else x)
    d = [12500 if x < 1000 else x for x in preds.salary]
    preds.salary = preds.salary.apply(lambda x: 350000 if x > 350000 else x)
    d1 = [350000 if x > 350000 else x for x in preds.salary]

    preds.to_csv('stacking_solution2.csv')

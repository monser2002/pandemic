import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error, make_scorer

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

import math

def rmsle1(y_true, y_pred):
    errors = list()
    for true, pred in zip(y_true, y_pred):
        if true < 0 or pred < 0: #check for negative values
            continue
        p = math.log(pred + 1)
        r = math.log(true + 1)
        errors.append((r - p)**2)
    return np.sqrt(np.mean(errors))

def rmsle(y_true, y_pred):
    
    return (np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1)))

train_a = pd.read_csv('train_region_group_a.csv', encoding='utf8', index_col='id')
train_b = pd.read_csv('train_region_group_b.csv', encoding='utf8', index_col='id')
train_c = pd.read_csv('train_region_group_c.csv', encoding='utf8', index_col='id')
train_d = pd.read_csv('train_region_group_d.csv', encoding='utf8', index_col='id')

X_a = pd.read_csv('x_a.csv', index_col='id')
X_b = pd.read_csv('x_b.csv', index_col='id')
X_c = pd.read_csv('x_c.csv', index_col='id')
X_d = pd.read_csv('x_d.csv', index_col='id')


my_scorer = make_scorer(rmsle, greater_is_better=False)

gbm_a = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
gbm_a.loss_ = my_scorer
gbm_a.fit(X_a, train_a.salary)


gbm_b = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
gbm_b.loss_ = my_scorer
gbm_b.fit(X_b, train_b.salary)


gbm_c = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
gbm_c.loss_ = my_scorer
gbm_c.fit(X_c, train_c.salary)


gbm_d = GradientBoostingRegressor(n_estimators=100, max_depth=5, max_features='auto', verbose=2)
gbm_d.loss_ = my_scorer
gbm_d.fit(X_d, train_d.salary)



test_a = pd.read_csv('test_a.csv', index_col='id')
test_b = pd.read_csv('test_b.csv', index_col='id')
test_c = pd.read_csv('test_c.csv', index_col='id')
test_d = pd.read_csv('test_d.csv', index_col='id')

preds_a = pd.DataFrame(gbm_a.predict(test_a), index=test_a.index, columns=['salary'])
preds_b = pd.DataFrame(gbm_b.predict(test_b), index=test_b.index, columns=['salary'])
preds_c = pd.DataFrame(gbm_c.predict(test_c), index=test_c.index, columns=['salary'])
preds_d = pd.DataFrame(gbm_d.predict(test_d), index=test_d.index, columns=['salary'])

preds = pd.concat([preds_a,preds_b,preds_c,preds_d])

preds.salary = preds.salary.apply(lambda x: 12500 if x < 1000 else x)
preds.salary = preds.salary.apply(lambda x: 350000 if x > 350000 else x)

preds.to_csv('gbm_solution2.csv')
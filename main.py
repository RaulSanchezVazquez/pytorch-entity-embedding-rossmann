#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:24:12 2018

@author: raulsanchez
"""

        
import pandas as pd
import datasets
import eval_utils
import numpy as np

import EntEmbNN as eenn
import xgboost as xgb

X_train, y_train, X_test, y_test = datasets.get_X_train_test_data()
dtrain = xgb.DMatrix(
    X_train.apply(lambda x: x.cat.codes),
    label=np.log(y_train))
evallist = [(dtrain, 'train')]
param = {'nthread': 12,
         'max_depth': 7,
         'eta': 0.02,
         'silent': 1,
         'objective': 'reg:linear',
         'colsample_bytree': 0.7,
         'subsample': 0.7}

num_round = 3000
bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

xgb_test_y_pred = bst.predict(
    xgb.DMatrix(X_test.apply(lambda x: x.cat.codes))
)
xgb_test_y_pred = np.exp((xgb_test_y_pred))


X_train, y_train, X_test, y_test = datasets.get_X_train_test_data()
for data in [X_train, X_test]:
    data.drop('Open', inplace=True, axis=1)

models = []
for _ in range(5):
    m = eenn.EntEmbNNRegression(
        cat_emb_dim={
            'Store': 10,
            'DayOfWeek': 6,
            'Promo': 1,
            'Year': 2,
            'Month': 6,
            'Day': 10,
            'State': 6},
        alpha=0,
        epochs=6,
        dense_layers = [1000, 500],
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        loss_function='L1Loss',
        train_size = .0,
        y_max = max(y_train.max(), y_test.max()))
    
    m.fit(X_train, y_train)
    models.append(m)
    print('\n')

test_y_pred = np.array([model.predict(X_test) for model in models])
test_y_pred = test_y_pred.mean(axis=0)

print("Mean Absolute Percentage Error")
print('XGB MAPE: %s' % eval_utils.MAPE(
    y_true=y_test, 
    y_pred=xgb_test_y_pred))

print('Ent.Emb. Neural Net MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:18:03 2018

@author: raulsanchez
"""

import pandas as pd
import datasets
import eval_utils
import numpy as np

import EENN as eenn

X, y, X_test, y_test = datasets.get_X_train_test_data()

for data in [X, X_test]:
    data.drop('Open', inplace=True, axis=1)

m = eenn.EntEmbNNRegression(
    cat_emb_dim={
        'Store': 10,
        'DayOfWeek': 6,
        'Promo': 1,
        'Year': 2,
        'Month': 6,
        'Day': 10,
        'State': 6},
    alpha=1e-5,
    epochs=10,
    dense_layers = [1000, 500],
    drop_out_layers = [0.0, 0.0],
    drop_out_emb = 0.0,
    loss_function='L1Loss',
    train_size=.9)

m.fit(X, y)

test_y_pred = m.predict(X_test)
print('MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))
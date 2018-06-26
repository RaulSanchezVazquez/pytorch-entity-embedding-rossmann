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
        alpha=1e-3,
        epochs=12,
        dense_layers = [1000, 500],
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        loss_function='L1Loss',
        train_size = 1.,
        y_max = max(y_train.max(), y_test.max()))
    
    m.fit(X_train, y_train)
    models.append(m)
    print('\n')

test_y_pred = np.array([model.predict(X_test) for model in models])
test_y_pred = test_y_pred.mean(axis=0)

print("Mean Absolute Percentage Error")
print('Ent.Emb. Neural Net MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))

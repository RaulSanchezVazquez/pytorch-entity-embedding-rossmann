#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:17:56 2018

@author: raul
"""
import pandas as pd
import numpy as np

def MAPE(y_true, y_pred):
    '''
    mean absolute percentage error
    '''
    
    abs_err = np.absolute((y_true - y_pred))
    percent_err = abs_err / y_true
    mape = np.sum(percent_err) / len(y_true)
    
    return mape

def RMSPE(y_true, y_pred):
    '''
    Root Mean Square Percentage Error (RMSPE).
    '''
    square_percent_err = ((y_true - y_pred) / y_true) ** 2
    mean_square_percent_err = pd.Series(square_percent_err).mean()
    
    rmspe = np.sqrt(mean_square_percent_err)
    
    return rmspe
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:17:56 2018

@author: raul
"""

import pandas as pd
import numpy as np
from  sklearn import metrics

def gini(solution, submission):
    '''
    '''                                 
    
    df = sorted(zip(solution, submission), key=lambda x : (x[1], x[0]),  reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def gini_norm(y_pred, y_true):
    '''
    '''
    
    normalized_gini = gini(y_pred, y_true)/gini(y_true, y_true)
    return normalized_gini    

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

def eval_regression(y_true, y_pred):
    reg_metrics = {
        'mean_absolute_error': metrics.mean_absolute_error(
            y_true=y_true, 
            y_pred=y_pred),
        'explained_variance_score': metrics.explained_variance_score(
            y_true=y_true, 
            y_pred=y_pred),
        'mean_squared_error': metrics.mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred),
        'median_absolute_error': metrics.median_absolute_error(
            y_true=y_true, 
            y_pred=y_pred),
        'r2_score': metrics.r2_score(
            y_true=y_true, 
            y_pred=y_pred),
        'gini_normalized': gini_norm(
            y_true=y_true, 
            y_pred=y_pred),
        'mean_absolute_percentage_error': MAPE(
            y_true=y_true, 
            y_pred=y_pred)
        }
        
    return pd.Series(reg_metrics)
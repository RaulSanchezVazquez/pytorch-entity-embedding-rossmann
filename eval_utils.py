#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:54:50 2018

@author: raulsanchez
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

def RMSLE(y_true, y_pred): 
    """
    Root Mean Squared Logarithmic Error
    """
    n = y_true.shape[0]
    
    return np.sqrt(
        (1 /n)  * (
            ( np.log1p(y_pred) - np.log1p(y_true) ) ** 2
        ).sum()
        )
    
    
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
            y_pred=y_pred),
        'root_mean_squared_logarithmic_error': RMSLE(
            y_true=y_true, 
            y_pred=y_pred)
        }
        
    return pd.Series(reg_metrics)

    
def classification_report(y_true, y_pred, y_score=None, average='micro'):
    '''
    Params:
    --------
    y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.
    
    y_score : nd array-like with the probabilities of the classes.
    
    average : str. either 'micro' or 'macro', for more details
        of how they are computed see:
            http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings
    Return:
    --------
    pd.DataFrame : contains the classification report as pandas.DataFrame 
    
    Example:
    ---------
    from sklearn.metrics import classification_report
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=5000, n_features=10,
                               n_informative=5, n_redundant=0,
                               n_classes=10, random_state=0, 
                               shuffle=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    
    sk_report = classification_report(
        digits=6,
        y_true=y_test, 
        y_pred=model.predict(X_test))
    
    report_with_auc = class_report(
        y_true=y_test, 
        y_pred=model.predict(X_test), 
        y_score=model.predict_proba(X_test))
    
    print(sk_report)
    
    Out:
             precision    recall  f1-score   support

          0   0.267101  0.645669  0.377880       127
          1   0.361905  0.290076  0.322034       131
          2   0.408451  0.243697  0.305263       119
          3   0.345455  0.327586  0.336283       116
          4   0.445652  0.333333  0.381395       123
          5   0.413793  0.095238  0.154839       126
          6   0.428571  0.474820  0.450512       139
          7   0.446809  0.169355  0.245614       124
          8   0.302703  0.466667  0.367213       120
          9   0.373333  0.448000  0.407273       125

    avg / total   0.379944  0.351200  0.335989      1250
        
    print(report_with_auc)
    
    Out:
                precision    recall  f1-score  support    pred       AUC
    0             0.267101  0.645669  0.377880    127.0   307.0  0.810550
    1             0.361905  0.290076  0.322034    131.0   105.0  0.777579
    2             0.408451  0.243697  0.305263    119.0    71.0  0.823277
    3             0.345455  0.327586  0.336283    116.0   110.0  0.844390
    4             0.445652  0.333333  0.381395    123.0    92.0  0.811389
    5             0.413793  0.095238  0.154839    126.0    29.0  0.654790
    6             0.428571  0.474820  0.450512    139.0   154.0  0.876458
    7             0.446809  0.169355  0.245614    124.0    47.0  0.777237
    8             0.302703  0.466667  0.367213    120.0   185.0  0.799735
    9             0.373333  0.448000  0.407273    125.0   150.0  0.825959
    avg / total   0.379944  0.351200  0.335989   1250.0  1250.0  0.800534
    
    '''
    
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return
    
    lb = LabelBinarizer()
    
    if len(y_true.shape) == 1:
        lb.fit(y_true)
    
    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    all_labels = set(labels).union(np.unique(y_true))
    
    pred_cnt = pd.Series(cnt, index=labels)
    pred_cnt = pred_cnt.loc[
        all_labels
    ].fillna(0)
    
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(all_labels))

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=all_labels)
    
    for l in (all_labels - set(class_report_df.columns)):
        class_report_df[l] = 0
    
    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]
    
    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total
    
    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])
            
            roc_auc[label] = auc(fpr[label], tpr[label])
        
        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())
            
            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])
        
        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))
            
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            
            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])
        
        class_report_df['AUC'] = pd.Series(roc_auc)
    
    return class_report_df
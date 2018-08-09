#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:17:39 2018

@author: lsanchez
"""

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNet(nn.Module, BaseEstimator, RegressorMixin):
    '''
    Parameters
    ----------
    cat_emb_dim : dict
        Dictionary containing the embedding sizes.
    
    layers : list
        NN. Layer arquitecture
    drop_out_layers : dict
        Dictionary with layer dropout
    drop_out_emb : float
        embedding drop out
    batch_size : int
        Mini Batch size
    val_idx : list
        
    allow_cuda : bool
    
    act_func : string
    
    lr : float
    
    alpha : float
    
    epochs : int
    '''
    def __init__(
        self,
        act_func='relu',        
        train_size=.8,
        batch_size=128,
        random_seed=None,
        verbose=True):
        
        super(NeuralNet, self).__init__()
        
        # General
        self.act_func = act_func
        self.train_size = train_size
        self.batch_size = int(batch_size)
        self.verbose = verbose
        self.random_seed = random_seed
        
        if not(self.random_seed is None):
            torch.manual_seed(self.random_seed) 
        
    def activ_func(self, x):
        '''
        Applies an activation function
        '''
        
        act_funcs = {
            'relu': F.relu, 
            'selu': F.selu}
        
        return act_funcs[self.act_func](x)
    
    def make_dataloader(self, X, y=None, shuffle=False, num_workers=8):
        '''
        Wraps a dataloader to iterate over (X, y)
        '''
        
        kwargs = {}
        if self.allow_cuda:
            kwargs = {'num_workers': 4, 'pin_memory': True}
        else:
            kwargs = {'num_workers': 4}
        
        if y is None:
            y = pd.Series([0] * X.shape[0])
        
        X, y = check_X_y(X, y.values.ravel())
        X = pd.DataFrame(X)
        y = pd.Series(y)
                
        loader = data_utils.DataLoader(
            data_utils.TensorDataset(
                torch.from_numpy(X.values).float(),
                torch.from_numpy(y.values).float()
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
            **kwargs)
        
        return loader
    
    def split_train_test(self):
        '''
        Splits Train-Test partitions
        '''
        
        err_msg = 'X size %s does not match y size %s'
        assert self.X.shape[0] == self.y.shape[0], err_msg % (
            self.X.shape, self.y.shape)
        
        if (self.train_size < 1) and (self.train_size > 0):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=self.train_size)
        else:    
            X_train = self.X
            X_test = self.X
            y_train = self.y
            y_test = self.y
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
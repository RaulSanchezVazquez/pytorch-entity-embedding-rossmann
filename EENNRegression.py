#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:25:31 2018

@author: lsanchez
"""

from EENN import EntEmbNN
import eval_utils


class EntEmbNNRegression(EntEmbNN):
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
        cat_emb_dim = {},
        dense_layers = [1000, 500],
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        act_func = 'relu',
        loss_function='MSELoss',
        train_size=1.,
        batch_size=128,
        epochs=10,
        lr=0.001,
        alpha=0.0,
        rand_seed=1,
        allow_cuda=False,
        random_seed=None,
        output_sigmoid=False,
        verbose=True):
        
        super(EntEmbNNRegression, self).__init__()
        
        # Model specific params.
        self.cat_emb_dim = cat_emb_dim
        self.dense_layers = dense_layers
        self.output_sigmoid = output_sigmoid
        
        # Reg. parameters
        self.drop_out_layers = drop_out_layers
        self.drop_out_emb = drop_out_emb
        self.alpha = alpha
        
        # Training params
        self.act_func = act_func
        self.train_size = train_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss_function = loss_function
        
        # Misc
        self.allow_cuda = allow_cuda
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Internal        
        self.embeddings = {}
        self.train_loss = []
        self.epochs_reports = []
        
        self.labelencoders = {}
        self.scaler = None
        
        self.num_features = None
        self.cat_features = None
        self.layers = {}
        
    def predict(self, X):
        """
        """
        
        return self.predict_raw(X)
    
    def eval_model(self):
        '''
        Model evaluation
        '''
        
        self.eval()
        
        test_y_pred = self.predict(self.X_test)
        
        report = eval_utils.eval_regression(
            y_true=self.y_test,
            y_pred=test_y_pred)
        
        msg = "\t[%s] Test: MSE:%s MAE: %s gini: %s R2: %s MAPE: %s"
        
        msg_params = (
            self.epoch_cnt, 
            round(report['mean_squared_error'], 6),
            round(report['mean_absolute_error'], 6),
            round(report['gini_normalized'], 6),
            round(report['r2_score'], 6),
            round(report['mean_absolute_percentage_error'], 6))
        
        self.epochs_reports.append(report)
        
        if self.verbose:
            print(msg % (msg_params))

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:25:31 2018

@author: lsanchez
"""

import numpy as np

from EENN import EntEmbNN
import eval_utils

class EntEmbNNBinary(EntEmbNN):
    '''
    Parameters
    ----------
    cat_emb_dim : dict
        { feature_name: int(emb_size) }
    dense_layers : list
        NN. architecture
    drop_out_layers : list
        Dropout in layers
    drop_out_emb : float
        Dropout in embeddings
    act_func : str
        Activation function {'relu', 'selu'}
    loss_function : str
        Pytorch loss-name
    train_size : float
    batch_size : int
    epochs : int
    lr : float
    alpha : float
    rand_seed : int
    allow_cuda : bool
    random_seed : int
    verbose : bool
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
        verbose_epoch=100,
        verbose=True):

        super(EntEmbNN, self).__init__()

        # Model specific params.
        self.cat_emb_dim = cat_emb_dim
        self.dense_layers = dense_layers

        output_sigmoid = True,
        self.output_sigmoid = output_sigmoid

        # Reg. parameters
        self.drop_out_layers = drop_out_layers
        self.drop_out_emb = drop_out_emb
        self.alpha = alpha

        # Training params
        self.act_func = act_func
        self.train_size = train_size
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = lr
        self.loss_function = loss_function

        # Misc
        self.allow_cuda = allow_cuda
        self.verbose = verbose
        self.verbose_epoch = verbose_epoch
        self.random_seed = random_seed

        # Internal
        self.embeddings = {}
        self.train_loss = []
        self.epochs_reports = []
        self.train_epoch_loss = []

        self.labelencoders = {}
        self.scaler = None

        self.num_features = None
        self.cat_features = None
        self.layers = {}

        # Model specific params.
        self.cat_emb_dim = cat_emb_dim
        self.dense_layers = dense_layers

    def predict(self, X):
        """
        """

        y_proba = self.predict_raw(X)
        y_pred = (y_proba > .5).astype(int)

        return y_pred

    def predict_proba(self, X):
        """
        """

        y_proba = self.predict_raw(X)
        y_proba = np.vstack([
            1 - y_proba,
            y_proba]).T

        return y_proba

    def eval_model(self):
        '''
        Model evaluation
        '''

        self.eval()

        y_proba = self.predict_proba(self.X_test)
        y_pred = (y_proba[:, 1] > .5).astype(int)

        report = eval_utils.classification_report(
            y_true=self.y_test,
            y_pred=y_pred,
            y_score=y_proba)


        msg = "\t[%s] Test: precision:%s recall: %s f1: %s auc: %s pred: [0] %s / %s [1] %s / %s"

        msg_params = (
            self.epoch_cnt,
            round(report['precision'].min(), 3),
            round(report['recall'].min(), 3),
            round(report['f1-score'].min(), 3),
            round(report['AUC'].iloc[0], 3),
            int(report['support'][0]),
            int(report['pred'][0]),
            int(report['support'][1]),
            int(report['pred'][1]))

        self.epochs_reports.append(report)

        if self.verbose:
            print(msg % (msg_params))

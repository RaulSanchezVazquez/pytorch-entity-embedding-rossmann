#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:25:31 2018

@author: lsanchez
"""


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.manifold import TSNE

import eval_utils
from base_neural_net import NeuralNet

class EntEmbNNRegression(NeuralNet):
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
        drop_out_layers = [0.5, 0.5],
        drop_out_emb = 0.2,
        act_func = 'relu',
        loss_function='SmoothL1Loss',
        train_size = .8,
        y_max=None,
        batch_size = 128,
        epochs=10,
        lr=0.001,
        alpha=0.0,
        rand_seed=1,
        allow_cuda=False,
        random_seed=None,
        verbose=True):
        
        super(EntEmbNNRegression, self).__init__()
        
        # Model specific params.
        self.cat_emb_dim = cat_emb_dim
        self.dense_layers = dense_layers
        self.y_max = y_max
        
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
        
        self.default_nan = None
        self.y_min = None
        self.ly_max = None
        
    def get_loss(self):
        if self.loss_function == 'SmoothL1Loss':
            return torch.nn.SmoothL1Loss()
        elif self.loss_function == 'L1Loss':
            return torch.nn.L1Loss()
        else:
            return torch.nn.MSELoss()
        
    def init_embeddings(self):
        '''
        Initializes the embeddings
        '''
        
        # Get embedding sizes from categ. features
        for f, le in self.labelencoders.items():
            #If no hand-set a dimension for the emb. set default
            
            if not f in self.cat_emb_dim:
                emb_dim = min(50, (len(le.classes_)) // 2 )
                self.cat_emb_dim[f] = emb_dim
            
            emb_dim = self.cat_emb_dim[f]
            
            self.embeddings[f] = nn.Embedding(
                len(le.classes_),
                emb_dim)
            
            # Weight initialization as original paper
            torch.nn.init.uniform_(
                self.embeddings[f].weight.data,
                a=-.05, b=.05)
            
            # Add emb. layer to model
            self.add_module(
                '[Emb %s]' % f, 
                self.embeddings[f])
    
    def init_dense_layers(self):
        '''
        Initializes dense layers
        '''
        
        input_size = (
            # Numb of Embedding neurons in input layer
            sum([sz for f, sz in self.cat_emb_dim.items()])
        ) + (
            # Numb of regular neurons for numerical features
            len(self.num_features)
        ) 
        
        NN_arquitecture = (
            [input_size]
        ) + (
            self.dense_layers
        ) + (
            [1]
        )
        
        for layer_idx, current_layer_size in enumerate(NN_arquitecture[:-1]):
            next_layer_size = NN_arquitecture[layer_idx + 1]
            
            layer_name = 'l%s' % (layer_idx + 1)
            layer = nn.Linear(current_layer_size, next_layer_size)
            
            self.add_module(layer_name, layer)
            
            self.layers[layer_name] = layer
    
    def X_fit(self, X):
        """
        """
        # Identify categorical vs numerical features
        self.cat_features = X.select_dtypes(include=['category']).columns
        self.num_features = X.select_dtypes(exclude=['category']).columns
        
        # Create encoders for categorical features
        self.labelencoders = {}
        for c in self.cat_features:
            le = LabelEncoder()
            le.fit(X[c].cat.codes + 1)
            self.labelencoders[c] = le
        
        if len(self.num_features) > 0:
            # Create scaler for numeric features
            self.scaler = MinMaxScaler()
            
            # Set default nan as 2 times the greatest value
            self.default_nan = X[self.num_features].max() * 2
            
            for c in self.num_features:
                X[c].fillna(self.default_nan.loc[c], inplace=True)
            
            self.scaler.fit(X[self.num_features])
    
    def X_transform(self, X):
        """
        X = X_test_nn.copy()
        """
        X = X.copy()
        for c in self.cat_features:
            codes = X[c].cat.codes + 1
            is_unknown_codes = ~codes.isin(
                self.labelencoders[c].classes_
            )
            codes[is_unknown_codes] = 0
            X[c] = self.labelencoders[c].transform(
                codes
            )
            
        if len(self.num_features) > 0:
            for c in self.num_features:
                X[c].fillna(self.default_nan[c], inplace=True)
        
            X = pd.concat([
                X[self.cat_features],
                pd.DataFrame(
                        self.scaler.transform(X[self.num_features]),
                        index=X.index,
                        columns=self.num_features)
                ], axis=1)
                
        return X
    
    def X_emd_replace(self, data):
        '''
        Returns the formated X-input, which is composed by the categorical
        embeddings and the respective continuous inputs.
        '''
        
        ''' Replace embeddings '''
        data_emb = []
        for f_idx, f in enumerate(self.cat_features):
            # Get column feature
            f_data = data[:, f_idx]
            
            if self.allow_cuda:
                f_data = f_data.cuda()

            # Retrieves the embeddings
            emb_cat = self.embeddings[f](f_data.long())
            
            #Apply Dropout
            emb_cat = F.dropout(
                emb_cat, 
                p=self.drop_out_emb,
                training=self.training)
            
            data_emb.append(emb_cat)
        
        ''' By-pass numeric '''
        if len(self.num_features) > 0:
            data_emb.append(
                data[:, len(self.cat_features):]
            )
        
        return torch.cat(data_emb, 1)
    
    def y_fit(self, y):
        """
        """
        self.y_min = y.min()
        if self.y_max is None:
            self.y_max = y.max()
        
        if self.y_min > 0:
            self.ly_max = np.log(self.y_max)
        else:
            self.ly_max = np.log1p(self.y.max)
            
        
    def y_transform(self, y):
        """
        """
        if self.y_min > 0:
            return np.log(y) / self.ly_max
        else:
            return np.log1p(y) / self.ly_max
        
    def y_transform_inverse(self, y):
        """
        """
        if self.y_min > 0:
            return np.exp(y * self.ly_max)
        else:
            return np.exp(y * self.ly_max) - 1
    
    def fit(self, X, y):
        '''
        X, y = X_train, y_train
        
        '''
        
        self.X = X.copy()
        self.y = y.copy()
        
        self.y_fit(self.y)
        self.X_fit(self.X)
        
        self.split_train_test()
        
        # Create embeddings and layers 
        self.init_embeddings()
        self.init_dense_layers()
        
        # GPU Flag
        if self.allow_cuda:
            self.cuda()
         
        self.iterate_n_epochs(epochs=self.epochs)
    
    def iterate_n_epochs(self, epochs):
        '''
        Makes N training iterations
        epochs = self.epochs
        '''
        
        self.epoch_cnt = 0
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.alpha
        )
        
        while(self.epoch_cnt < self.epochs):
            self.train()
            loss_func = self.get_loss()
            
            dataloader = self.make_dataloader(
                # Format X such as categ.first then numeric.
                self.X_transform(self.X_train),
                # Normalize such as log(y) / y_log_max
                self.y_transform(self.y_train)
                
            )
            
            for batch_idx, (x, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                if self.allow_cuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target).float()
                
                output = self.forward(x)
                
                loss = loss_func(
                    output.reshape(1, -1)[0],
                    target.float())
                
                # self.layers['l1'].weight.data
                loss.backward()
                self.optimizer.step()
                
                self.train_loss.append(loss.item())
            
            self.train_loss[:100]
            
            self.epoch_cnt += 1
            self.eval_model()
            
    def forward(self, x):
        '''
        Forward pass
        x_ = x
        '''
        
        # Parse batch with embeddings
        x = self.X_emd_replace(x)
        
        # Forward pass on dense layers
        n_layers = len(self.layers.items())
        for layer_idx in range(n_layers):
            layer_name = 'l%s' % (layer_idx + 1)
            layer = self.layers[layer_name]
            
            is_inner_layer = (
                layer_idx < len(self.dense_layers)
            )
            
            x = layer(x)
            
            # Do not apply act.func on last layer
            if is_inner_layer:
                
                # Apply dropout
                x = F.dropout(
                    x,
                    p=self.drop_out_layers[layer_idx],
                    training=self.training)
                
                x = self.activ_func(x)
            else:
                x = F.sigmoid(x)
        
        return x
    
    def predict(self, X):
        '''
        Predict scores
        
        self = NNmodel
        X  = self.X_test
        '''
        
        #Set pytorch model in eval. mode
        self.eval()
        
        dataloader = self.make_dataloader(self.X_transform(X))
        
        y_pred = []
        for batch_idx, (x, _) in enumerate(dataloader):
            if self.allow_cuda:
                x = x.cuda()
            x = Variable(x)
            
            output = self.forward(x)
            
            if self.allow_cuda:
                output = output.cpu()
            y_pred += output.data.numpy().flatten().tolist()
        
        y_pred = np.array(y_pred)
        y_pred = self.y_transform_inverse(y_pred)
        
        return y_pred
    
    def get_embeddings(self):
    
        embeddings = {}
        for c in self.cat_features:
            categ_names = self.X[c].drop_duplicates()
            categ_codes = categ_names.cat.codes
            categories = pd.Series(
                [x for x in categ_names], 
                index=categ_codes.values)
            categories.sort_index(inplace=True)
            categories.index = categories.index + 1
            
            emb = self.embeddings[c].weight.data
            if self.allow_cuda:
                emb = emb.cpu()
                
            emb = pd.DataFrame(
                emb.numpy(),
                index=categories.values)
            emb = emb.add_prefix('latent_')
            embeddings[c] = emb
            
        return embeddings
    
    def eval_model(self):
        '''
        Model evaluation
        '''
        
        self.eval()
        
        test_y_pred = self.predict(self.X_test)
        
        report = eval_utils.eval_regression(
            y_true=self.y_transform(self.y_test),
            y_pred=self.y_transform(test_y_pred))
        
        msg = "\t[%s] Test: MSE:%s MAE: %s gini: %s R2: %s MAPE: %s"
        
        msg_params = (
            self.epoch_cnt, 
            round(report['mean_squared_error'], 5),
            round(report['mean_absolute_error'], 5),
            round(report['gini_normalized'], 5),
            round(report['r2_score'], 5),
            round(report['mean_absolute_percentage_error'], 5))
        
        self.epochs_reports.append(report)
        
        if self.verbose:
            print(msg % (msg_params))

def test():
    import pandas as pd
    import datasets
    import eval_utils
    import numpy as np
    
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
    for random_seed in range(5):
        self = EntEmbNNRegression(
            cat_emb_dim={
                'Store': 10,
                'DayOfWeek': 6,
                'Promo': 1,
                'Year': 2,
                'Month': 6,
                'Day': 10,
                'State': 6},
            alpha=0,
            epochs=10,
            dense_layers = [1000, 500],
            drop_out_layers = [0., 0.],
            drop_out_emb = 0.0,
            loss_function='L1Loss',
            train_size = 1.,
            y_max = max(y_train.max(), y_test.max()),
            random_seed=random_seed)
        
        self.fit(X_train, y_train)
        models.append(self)
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

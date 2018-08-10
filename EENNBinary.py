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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.manifold import TSNE

import eval_utils
from base_neural_net import NeuralNet

class EntEmbNNBinary(NeuralNet):
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
        loss_function='BCELoss',
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
        
        super(EntEmbNNBinary, self).__init__()
        
        # Model specific params.
        self.cat_emb_dim = cat_emb_dim
        self.dense_layers = dense_layers
        
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
        
        self.num_features = None
        self.cat_features = None
        self.layers = {}
        
    def get_loss(self, loss_name):
        if loss_name == 'BCELoss':
            return torch.nn.BCELoss()
        elif loss_name == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.BCELoss()
        
    def init_embeddings(self):
        '''
        Initializes the embeddings
        '''
        
        # Get embedding sizes from categ. features
        for f in self.cat_features:
            le = self.labelencoders[f]
            
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
            sum([
                self.embeddings[f].weight.data.shape[1] 
                for f in self.cat_features
            ])
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
        self.cat_features = list(self.cat_emb_dim.keys())
        self.num_features = list(set(
            self.X.columns.tolist()
        ).difference(self.cat_features))
        
        # Create encoders for categorical features
        self.labelencoders = {}
        for c in self.cat_features:
            le = LabelEncoder()
            le.fit( X[c].astype(str).tolist())
            self.labelencoders[c] = le

    
    def X_transform(self, X):
        """
        X = X_test_nn.copy()
        """
        X = X.copy()
        for c in self.cat_features:
            codes = X[c].astype(str)
            X[c] = self.labelencoders[c].transform(
                codes
            )
        
        X = X[self.cat_features + self.num_features]
        
        return X
    
    def X_emd_replace(self, data):
        '''
        Returns the formated X-input, which is composed by the categorical
        embeddings and the respective continuous inputs.
        '''
        
        #data = x
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
        
        ''' Concat numeric features '''
        if len(self.num_features) > 0:
            data_emb.append(
                data[:, len(self.cat_features):]
            )
        
        return torch.cat(data_emb, 1)
    
    def fit(self, X, y):
        """
        """
        
        self.X = X.copy()
        self.y = y.copy()
        
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
            loss_func = self.get_loss(self.loss_function)
            
            dataloader = self.make_dataloader(
                # Format X such as categ.first then numeric.
                self.X_transform(self.X_train),
                self.y_train
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
                
                loss.backward()
                self.optimizer.step()
                
                self.train_loss.append(loss.item())
            
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
                x = torch.sigmoid(x)
        
        return x
    
    def predict_proba(self, X):
        """ 
        """
        y_pred = self.predict_raw(X)
        y_score = np.vstack([
            1 - y_pred,
            y_pred]).T
        
        return y_score
    
    def predict(self, X):
        """
        """
        
        y_pred = self.predict_raw(X)
        y_pred = (y_pred > .5).astype(int)
        
        return y_pred
        
    def predict_raw(self, X):
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
        
        test_y_score = self.predict_proba(self.X_test)
        test_y_pred = (test_y_score[:, 1] > .5).astype(int)

        report = eval_utils.classification_report(
            y_true=self.y_test.values,
            y_pred=test_y_pred,
            y_score=test_y_score)
        
        self.epochs_reports.append(report)
        
        if self.verbose:
            print(report)

def test():
    import pandas as pd
    import datasets
    import eval_utils
    import numpy as np
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    X, y, X_test, y_test = datasets.get_X_train_test_data()
    
    for data in [X, X_test]:
        data.drop('Open', inplace=True, axis=1)
    
    for data in [X, X_test]:
        for f in data.columns:
            data[f] = data[f].cat.codes
    
    
    y = np.log(y)
    y_test = np.log(y_test)
    
    quantile = y.quantile(.8)
    y = (y < quantile).astype(int)
    y_test = (y_test < quantile).astype(int)
    
    params = {
        'dense_layers':[100, 100],
        'act_func': 'relu',
        'alpha': 0.0001,
        'batch_size': 64,
        'lr': .001,
        'epochs': 10,
        'rand_seed': 1,
    }
    
    self = EntEmbNNBinary(
        cat_emb_dim={
                'Month': 6,
                'Store': 10,
                'Promo': 1,
                'Year': 2,
                'DayOfWeek': 6,
                'Day': 10,
                'State': 6},
        dense_layers = params['dense_layers'],
        act_func =params['act_func'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        epochs=params['epochs'],
        rand_seed=params['rand_seed'],
        
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        loss_function='BCELoss',
        train_size=.8,
        allow_cuda=False,
        verbose=True)
    
    self.fit(X, y)
    
    nn_y_pred = self.predict(X_test)
    nn_y_score = self.predict_proba(X_test)
    print((eval_utils.classification_report(
            y_true=y_test, 
            y_pred=nn_y_pred,
            y_score=nn_y_score
        )).round(5))
    
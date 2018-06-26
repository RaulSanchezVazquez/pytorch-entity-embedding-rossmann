#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:51:34 2018

@author: raulsanchez
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

torch.manual_seed(1)

class EntEmbNNRegression(nn.Module):
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
        lr=0.001,
        epochs=10,
        alpha=0.0,
        layers = [1000, 500],
        #drop_out_layers = [0., 0.],
        #drop_out_emb = 0.,
        batch_size = 128,
        X_y_test = None,
        allow_cuda = False,
        act_func = 'relu',
        train_size = None,
        y_range = None,
        verbose=True):
        
        super(EntEmbNNRegression, self).__init__()
        
        #Dict contains the embedding
        self.cat_emb_dim = cat_emb_dim
        self.layers = layers
        #self.drop_out_layers = drop_out_layers
        #self.drop_out_emb = drop_out_emb
        self.batch_size = batch_size
        #self.val_idx = val_idx
        self.allow_cuda = allow_cuda
        self.act_func = act_func
        self.lr = lr
        self.alpha = alpha
        self.y_range = y_range
        self.epochs = epochs
        self.train_size = train_size
        self.X_y_test = X_y_test
        self.embeddings = {}
        self.dense_layers = {}
        self.cat_vocab_size = {}
        self.in_layer_size = 0
        self.global_loss = []
        self.verbose = verbose
        self.mb_log_loss = []
        
    def activ_func(self, x):
        '''
        Applies an activation function
        '''
        
        act_funcs = {
            'relu': F.relu, 
            'selu': F.selu}
        
        return act_funcs[self.act_func](x)
    
    def split_train_test(self):
        '''
        Splits Train-Test partitions
        '''
        #cols = self.X.columns
        
        _, _ = check_X_y(
                self.X, 
                self.y.values.ravel())
        
        if not(self.X_y_test is None):
            X_train = self.X
            y_train = self.y
            (X_test, y_test) = self.X_y_test
            
            X_test = self.format_X(X_test)
            y_test = self.format_y(y_test)
        
        elif not(self.train_size is None):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=self.train_size)
        else:
            X_test = self.X
            y_test = self.y
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    def make_dataloader(self, X, y=None, shuffle=False, num_workers=8):
        '''
        Wraps a dataloader to iterate over (X, y)
        '''
        
        kwargs = {}
        if self.allow_cuda:
            kwargs = {'num_workers': 1, 'pin_memory': True}
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
    
    def init_embeddings(self):
        '''
        Initializes the embeddings
        '''
        
#        self.cat_features_idx = [
#            self.X.columns.get_loc(x) 
#            for x in self.cat_features]
        
        #Get embedding sizes from categ. features
        self.in_layer_size += len(self.num_features)
        for f in self.cat_features:
            #Get vocabulary length
            vocab_size = self.X[f].nunique()
            self.cat_vocab_size[f] = vocab_size
            
            #If no hand-set a dimension for the emb. set default
            if not f in self.cat_emb_dim:
                emb_dim = min(50, (vocab_size + 1) // 2 )
                self.cat_emb_dim[f] = emb_dim
            else:
                emb_dim = self.cat_emb_dim[f]
            
            self.embeddings[f] = nn.Embedding(
                vocab_size,
                emb_dim)
            
            #Same initialization as original paper
            torch.nn.init.uniform_(
                self.embeddings[f].weight.data,
                a=-.05, b=.05)
            
            #Add emb. layer to model
            self.add_module(f'[Emb {f}]', self.embeddings[f])
            
            #sum. emb. size to obtain size of input layer
            self.in_layer_size += emb_dim
    
    def init_layers(self):
        '''
        Initializes dense layers
        '''
        
        net_arq = [self.in_layer_size] + self.layers + [1]
        for h_it, h in enumerate(net_arq[:-1]):
            hl_label = 'l%s'%(h_it+1)
            hl = nn.Linear(net_arq[h_it], net_arq[h_it+1])
            
            #Same initialization as original paper
#            torch.nn.init.uniform_(
#                hl.weight.data, 
#                a=-.5, b=5)
            
            self.add_module(hl_label, hl)
            self.dense_layers[hl_label] = hl
    
    def get_y_norm_constants(self):
        if self.y_range is None:
            self.y_max = self.y.max()
            self.y_min = self.y.min()
            
            if not(self.X_y_test is None):
                self.y_max = max(
                    self.X_y_test[1].max(),
                    self.y_max)
                
                self.y_min = min(
                    self.X_y_test[1].min(),
                    self.y_min)
        else:
            self.y_min, self.y_max = self.y_range
    
    def format_y(self, y):
        '''
        '''
        
        return np.log(y) / np.log(self.y_max)
    
    def format_y_inv(self, ylog):
        '''
        '''
        return np.exp(ylog * np.log(self.y_max))
    
    def format_X(self, X):
        '''
        Format X such as categorical features are found first
        on the X matrix, followed by the numericals.
        Also ensure that categories are encoded as their actual code
        and not their string.
        '''
        return pd.concat([
            X[self.cat_features].apply(lambda x: x.cat.codes),
            X[self.num_features]
            ], axis=1)
            
    def format_X_emb(self, data):
        '''
        Returns the formated X-input, which is composed by the categorical
        embeddings and the respective continuous inputs.
            
        '''
        
        #Contains all features to concat.
        x = []
        for f_it, f in enumerate(self.cat_features):
            #Get column feature
            cat_input = data[:, f_it]
            if self.allow_cuda:
                cat_input = cat_input.cuda()

            #Retrieves the embeddings
            emb_cat = self.embeddings[f](cat_input.long())
            
            #Apply Dropout
#            emb_cat = F.dropout(
#                emb_cat, 
#                p=self.drop_out_emb, 
#                training=self.training)
            
            #Store 
            x.append(emb_cat)
        
        #Get continious features    
        boundary_categ = len(self.cat_features)
        has_continious_features = data.shape[1] > boundary_categ
        if has_continious_features:
            x.append(data[:, boundary_categ:])
        
        #concat the embeddings and the continous features.
        x = torch.cat(x, 1)
        
        return x
    
    def forward(self, x):
        '''
        Forward pass
        x = data
        x.shape
        self
        '''
        
        #Parse batch with embeddings
        x = self.format_X_emb(x)
        
        #Forward pass on dense layers
        for hl_it, (hl_name, hl) in enumerate(self.dense_layers.items()):
            is_inner_layer = (
                hl_it <= (len(self.dense_layers) - 2)
            )
            x = hl(x)
            if is_inner_layer:
#                x = F.dropout(
#                    x,
#                    p=self.drop_out_layers[hl_it],
#                    training=self.training)
    
                x = self.activ_func(x)
            else:
                
                x = F.sigmoid(x)
        
        return x
    
    def fit(self, X, y):
        '''
        self = m
        
        '''
        
        self.X = X.copy()
        self.y = y.copy()
        
        #Identify categorical vs numerical features
        self.cat_features = X.select_dtypes(include=['category']).columns
        self.num_features = X.select_dtypes(exclude=['category']).columns
        
        self.X = self.format_X(self.X)
        self.get_y_norm_constants()
        
        self.y = self.format_y(self.y)
        
        #Create embeddings and layers 
        self.init_embeddings()
        self.init_layers()
        
        # self.embeddings['Store'].weight.data
        # self.dense_layers['l3'].weight.data
        
        #GPU Flag
        if self.allow_cuda:
            self.cuda()
         
        #Make Train/Test Splits
        self.split_train_test()
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr, 
            weight_decay=0)
        
        self.log_losses = []
        #self.epoch_cnt = 0
        self.iterate_n_epochs(epochs=self.epochs)
        
    def iterate_n_epochs(self, epochs):
        '''
        Makes N training iterations
        '''
        
        self.epoch_cnt = 0
        while(self.epoch_cnt < self.epochs):
            
            self.train()
            loss_func = torch.nn.L1Loss()
            
            #create data-loop object
            dataloader = self.make_dataloader(
                #Format X such as categ.first then numeric.
                self.X_train,
                #Normalize such as log(y) / y_log_max
                self.y_train
            )
            
            for batch_idx, (data, target) in enumerate(dataloader):
                
                self.optimizer.zero_grad()
                
                if self.allow_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target).float()
                
                output = self.forward(data)
                
                loss = loss_func(
                    output.reshape(1, -1)[0],
                    target.float())
                
                # self.dense_layers['l1'].weight.data
                loss.backward()
                self.optimizer.step()
                
                self.log_losses.append(loss.item())
            
            self.log_losses[:100]
            
            self.epoch_cnt += 1
            self.eval_model()
    
    def predict(self, X, log_scale=True, format_X=True):
        '''
        Predict scores
        X = self.X_train
        '''
        
        #Set pytorch model in eval. mode
        self.eval()
        if format_X:
            X = self.format_X(X)
        #create data-loop object
        dataloader = self.make_dataloader(
            X, #Format X such as categ.first then numeric.
            shuffle=False)
        
        y_pred = []
        for batch_idx, (data, _) in enumerate(dataloader):
            if self.allow_cuda:
                data = data.cuda()
            data = Variable(data)
            
            output = self.forward(data)
            
            if self.allow_cuda:
                output = output.cpu()
            y_pred += output.data.numpy().flatten().tolist()
        y_pred = np.array(y_pred)
        
        if log_scale:
            y_pred = self.format_y_inv(y_pred)
        
        return y_pred
    
    def eval_model(self):
        '''
        Model evaluation
        '''
        
        self.eval()
        
        train_y_pred = self.predict(
            self.X_train,
            log_scale=False, 
            format_X=False)
        
        self.format_y_inv(train_y_pred)
        
        loss_train = pd.Series(
            self.y_train - train_y_pred
        ).abs().mean()
        
        test_y_pred = self.predict(
            self.X_test, 
            log_scale=False,
            format_X=False)
        
        loss_test = pd.Series(
            self.y_test - test_y_pred
        ).abs().mean()
        
        msg = "\t[%s] MAPE train:%s MAPE test:%s"
        
        msg_params = (
            self.epoch_cnt, 
            round(loss_train, 5),
            round(loss_test, 5))
        
        if self.verbose:
            print(msg % (msg_params))

def test():
    import pandas as pd
    import datasets
    import eval_utils
    import numpy as np    
    
    X, y, X_test, y_test = datasets.get_X_train_test_data()
    
    for data in [X, X_test]:
        data.drop('Open', inplace=True, axis=1)
    
    models = []
    for _ in range(5):
        m = EntEmbNNRegression(
            X_y_test = (X_test, y_test),
            cat_emb_dim={
                'Store': 10,
                'DayOfWeek': 6,
                'Promo': 1,
                'Year': 2,
                'Month': 6,
                'Day': 10,
                'State': 6})
    
        m.fit(X, y)
        models.append(m)
        print('\n')
    
    test_y_pred = np.array([model.predict(X_test) for model in models])
    test_y_pred = test_y_pred.mean(axis=0)
    
    print('MAPE: %s' % eval_utils.MAPE(
        y_true=y_test.values.flatten(),
        y_pred=test_y_pred))
    
    eval_utils.eval_regression(
        y_true=y_test, 
        y_pred=test_y_pred)
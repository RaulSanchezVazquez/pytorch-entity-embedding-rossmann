#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:25:31 2018

@author: lsanchez
"""


from EENN import EntEmbNN


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


def test_rossman():
    import pandas as pd
    import datasets
    import eval_utils
    import numpy as np
    
    X, y, X_test, y_test = datasets.get_X_train_test_data()
    
    for data in [X, X_test]:
        data.drop('Open', inplace=True, axis=1)
    
    for data in [X, X_test]:
        for f in data.columns:
            data[f] = data[f].cat.codes
    
    y = np.log(y) / np.log(41551)
    y_test = np.log(y_test) / np.log(41551)
    
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
            output_sigmoid=True,
            lr=0.001,
            train_size = 1.,
            random_seed=random_seed)
        
        self.fit(X, y)
        models.append(self)
        print('\n')
    
    test_y_pred = np.array([model.predict(X_test) for model in models])
    test_y_pred = test_y_pred.mean(axis=0)
    
    print('Ent.Emb. Neural Net MAPE: %s' % eval_utils.MAPE(
        y_true=np.exp(y_test.values.flatten() * np.log(41551)),
        y_pred=np.exp(test_y_pred * np.log(41551))))

def test_pure_neural_net_vs_sklearn():
    import pandas as pd
    import datasets
    import eval_utils
    import numpy as np
    
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    
    X, y, X_test, y_test = datasets.get_X_train_test_data()
    
    for data in [X, X_test]:
        data.drop('Open', inplace=True, axis=1)
    
    for data in [X, X_test]:
        for f in data.columns:
            data[f] = data[f].cat.codes
    
    scaler = MinMaxScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns)
    
    X_test = pd.DataFrame(
        scaler.fit_transform(X_test),
        columns=X_test.columns)
    
    y = np.log(y)
    y_test = np.log(y_test)

    params = {
        'dense_layers':[100, 100],
        'act_func': 'relu',
        'alpha': 0.0001,
        'batch_size': 64,
        'lr': .001,
        'epochs': 5,
        'rand_seed': 1,
    }
    
    eenn_model = EntEmbNNRegression(
        cat_emb_dim = {},
        dense_layers = params['dense_layers'],
        act_func =params['act_func'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        epochs=params['epochs'],
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        loss_function='MSELoss',
        train_size=1.,
        allow_cuda=False,
        verbose=True)
    
    eenn_model.fit(X, y)
    
    eenn_y_pred = eenn_model.predict(X_test)
    print((eval_utils.eval_regression(
            y_true=np.exp(y_test * np.log(41551)), 
            y_pred=np.exp(eenn_y_pred * np.log(41551))
        )).round(5))
    
    sk_model = MLPRegressor(
        hidden_layer_sizes=params['dense_layers'],
        activation=params['act_func'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        learning_rate_init=params['lr'],
        max_iter=params['epochs'],
        random_state=params['rand_seed'],

        solver='adam',
        learning_rate='constant',
        validation_fraction=0.,
        verbose=True,
        momentum=False,
        early_stopping=False,
        epsilon=1e-8)
    
    sk_model.fit(X, y)
    sk_y_pred = sk_model.predict(X_test)
    
    print((eval_utils.eval_regression(
            y_true=np.exp(y_test * np.log(41551)), 
            y_pred=np.exp(sk_y_pred * np.log(41551))
        )).round(5))

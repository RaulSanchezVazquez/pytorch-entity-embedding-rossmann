#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:57:36 2018

@author: raul

NOTE: hand-configure DATAPATH var. in this script to the folder
where files will be downloaded.

Original codes from:
    https://github.com/entron/entity-embedding-rossmann/blob/master/extract_csv_files.py
    https://github.com/entron/entity-embedding-rossmann/blob/master/prepare_features.py
    https://github.com/entron/entity-embedding-rossmann/blob/master/train_test_model.py
"""

import os
import csv
import random
import pandas as pd
import numpy as np
import requests
import subprocess

DATAPATH = os.path.expanduser('~/data_rossmann')
ROSSMANN_PATH = os.path.join(DATAPATH, 'rossmann.tgz')
random.seed(42)
np.random.seed(123)


def download_rossmann():
    '''
    Code from:
    https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
    
    Thanks to FAST.AI to host the dataset for us!
    '''
    
    #Thanks for fast.ai to host the dataset.
    url = 'http://files.fast.ai/part2/lesson14/rossmann.tgz'
    try:
        if not os.path.exists(DATAPATH):
            os.mkdir(DATAPATH)
            
        r = requests.get(url, stream=True)
        output_file = open(ROSSMANN_PATH, 'wb')
        
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                output_file.write(chunk)
        output_file.close()
        
        subprocess.call(f'tar xvf {ROSSMANN_PATH} -C {DATAPATH}'.split(' '))
    except:
        print(
            "Download Failed, "
            "please manually download data in "
            f"{DATAPATH}")
        
def csv2dicts(csvfile):
    '''
    Reads csv and store each line as json in
    the format:
        ---------
        csv file:
        ---------
        colname1, colname2
        val1, val2
        val3, val4
        .
        .
        .
        
        -------
        Output:
        -------
        {'colname1': val1, 'colname2': val2}
        {'colname1': val3, 'colname2': val4}
        .
        .
        .
        
    '''
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            keys = row
            continue
        
        data.append({key: value for key, value in zip(keys, row)})
        
    return data

def set_nan_as_string(data, replace_str='0'):
    '''
    Replaces '' with the '0' string.
    Original code in:
        https://github.com/entron/entity-embedding-rossmann/blob/master/extract_csv_files.py
    '''
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x

def get_X_train_test_data(simulate_sparsity=True):
    '''
    Computes the dataset used in:
    
    @article{guo2016entity,
      title={Entity embeddings of categorical variables},
      author={Guo, Cheng and Berkhahn, Felix},
      journal={arXiv preprint arXiv:1604.06737},
      year={2016}
    }
    
    After first computation files are saved and cached as HDF.
    
    Return:
    -------
    X_train : pandas.DataFrame
    y_train : pandas.Series
    X_test : pandas.DataFrame
    y_test : pandas.Series
    '''
    
    if not os.path.exists(ROSSMANN_PATH):
        download_rossmann()
    
    if simulate_sparsity:
        dataset_output_path = f'{DATAPATH}/X_train_test_sparse.hdf'
    else:
        dataset_output_path = f'{DATAPATH}/X_train_test_no_sparse.hdf'
    
    if not os.path.exists(dataset_output_path):
        train_data_path = f"{DATAPATH}/train.csv"
        with open(train_data_path) as csvfile:
            train_data = csv.reader(csvfile, delimiter=',')
            train_data = csv2dicts(train_data)
            train_data = train_data[::-1]
        
        store_data_path = f"{DATAPATH}/store.csv"
        with open(store_data_path) as csvfile:
            store_data = csv.reader(csvfile, delimiter=',')
            store_data = csv2dicts(store_data)
        
        store_states_path = f"{DATAPATH}/store_states.csv"    
        with open(store_states_path) as csvfile2:
            state_data = csv.reader(csvfile2, delimiter=',')
            state_data = csv2dicts(state_data)
            
        set_nan_as_string(store_data)
        for index, val in enumerate(store_data):
            state = state_data[index]
            val['State'] = state['State']
            store_data[index] = val
        
        '''
        prepare_features.py
        '''
        train_data = pd.DataFrame(train_data)
        is_open_defined = train_data['Open'] != ''
        has_sales = train_data['Sales'] != '0'
        
        train_data = train_data[(has_sales) & (is_open_defined)]
        
        '''
        The following lines does what the function:
            
            feature_list(record, store_data)
            
        used to do in script prepare_features.py, but in pandas-like
        operations.
        '''
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        train_data['Store'] = train_data['Store'].astype(int)
        train_data['Year'] = train_data['Date'].dt.year.values
        train_data['Month'] = train_data['Date'].dt.month.values
        train_data['Day'] = train_data['Date'].dt.day.values
        train_data['DayOfWeek'] = train_data['DayOfWeek'].astype(int)
        train_data['Open'] = train_data['Open'].astype(int)
        train_data['Promo'] = train_data['Promo'].astype(int)
        train_data['State'] = train_data['Store'].apply(
            lambda x: store_data[x -1]['State'])
        
        cols = ['Open', 'Store', 'DayOfWeek', 'Promo', 
                'Year', 'Month', 'Day', 'State']
        train_data_X = train_data[cols]
        
        train_data_y = train_data['Sales'].astype(int)
        
        for c in train_data_X.columns:
            train_data_X[c] = train_data_X[c].astype(
                'category').cat.as_ordered()
        
        '''
        Make train/test splits
        '''
        
        train_ratio = 0.9
        train_size = int(train_ratio * train_data_X.shape[0])
        
        X_train = train_data_X[:train_size]
        y_train = train_data_y[:train_size]

        X_test = train_data_X[train_size:]
        y_test = train_data_y[train_size:]

        X_train.head(10)
        X_train.tail(10)
        # Simulate data sparsity
        if simulate_sparsity:
            size = 200000
            
            idx = np.random.randint(
                X_train.shape[0],
                size=size)
            
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
        
        
        X_train.to_hdf(
            dataset_output_path, key='X_train', format="table")
        X_test.to_hdf(
            dataset_output_path, key='X_test', format="table")
        y_train.to_hdf(
            dataset_output_path, key='y_train', format="table")
        y_test.to_hdf(
            dataset_output_path, key='y_test', format="table")
    else:
        X_train = pd.read_hdf(
            dataset_output_path, key='X_train', format="table")
        X_test = pd.read_hdf(
            dataset_output_path, key='X_test', format="table")
        y_train = pd.read_hdf(
            dataset_output_path, key='y_train', format="table")
        y_test = pd.read_hdf(
            dataset_output_path, key='y_test', format="table")
    
    return X_train, y_train, X_test, y_test
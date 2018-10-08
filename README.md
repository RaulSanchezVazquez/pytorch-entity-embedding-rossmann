
# Entity-embedding-rossmann 

This is a Pytorch implementation with sklearn model interface for which most DS are familiar (`model.fit(X, y)`  and `model.predict(X, y)`)

This implementation reproduces the code used in the paper **"[Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737)"** and extends its functionality to other Machine Learning problems. 

The original Keras code used as a benchmark can be found in: 
**Entity Embeddings of Categorical Variables [REPO](https://github.com/entron/entity-embedding-rossmann)**.

# Notes

This repo aims to provide an Entity Embedding Neural Network out-of-the-box model for Regression and Classification tasks.

To this date this repo has implemented:

- Regression (tested on original implementation in here).
- Binary Classification (used `EntEmbNNBinary` instead of `EntEmbNNRegression`) (tested on personal projects).


```python
import pandas as pd
import datasets
import eval_utils
import numpy as np

from EENNRegression import EntEmbNNRegression

X, y, X_test, y_test = datasets.get_X_train_test_data()

# This normalization comes from original Entity Emb. original Code
y_max = max(y.max(), y_test.max())
y = np.log(y) / np.log(y_max)
y_test = np.log(y_test) / np.log(y_max)

for data in [X, X_test]:
    data.drop('Open', inplace=True, axis=1)
```


```python
models = []
for _ in range(5):
    m = EntEmbNNRegression(
        cat_emb_dim={
            'Store': 10,
            'DayOfWeek': 6,
            'Promo': 1,
            'Year': 2,
            'Month': 6,
            'Day': 10,
            'State': 6},
        alpha=0,
        dense_layers=[1000, 500],
        drop_out_layers=[0., 0.],
        drop_out_emb=0.,
        loss_function='L1Loss',
        train_size=1., 
        verbose=True)

    m.fit(X, y)
    models.append(m)
    print('\n')

test_y_pred = np.array([model.predict(X_test) for model in models])
test_y_pred = test_y_pred.mean(axis=0)

print('MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))
```

    	[1] Test: MSE:0.000197 MAE: 0.010257 gini: 0.901585 R2: 0.878147 MAPE: 0.012573
    	[2] Test: MSE:0.000156 MAE: 0.00897 gini: 0.890218 R2: 0.903692 MAPE: 0.011022
    	[3] Test: MSE:0.000146 MAE: 0.008785 gini: 0.916184 R2: 0.909749 MAPE: 0.010714
    	[4] Test: MSE:0.000157 MAE: 0.00908 gini: 0.890544 R2: 0.902871 MAPE: 0.011217
    	[5] Test: MSE:0.000132 MAE: 0.008313 gini: 0.918975 R2: 0.918285 MAPE: 0.010129
    	[6] Test: MSE:0.000122 MAE: 0.007781 gini: 0.921821 R2: 0.924628 MAPE: 0.009557
    	[7] Test: MSE:0.000157 MAE: 0.009234 gini: 0.907448 R2: 0.903 MAPE: 0.011397
    	[8] Test: MSE:0.000115 MAE: 0.007533 gini: 0.93635 R2: 0.929313 MAPE: 0.009253
    	[9] Test: MSE:0.000116 MAE: 0.007687 gini: 0.933982 R2: 0.928413 MAPE: 0.009381
    	[10] Test: MSE:0.000107 MAE: 0.0073 gini: 0.950313 R2: 0.933775 MAPE: 0.008926
    
    
    	[1] Test: MSE:0.000313 MAE: 0.013938 gini: 0.894436 R2: 0.807053 MAPE: 0.016888
    	[2] Test: MSE:0.00016 MAE: 0.009192 gini: 0.937388 R2: 0.901161 MAPE: 0.011281
    	[3] Test: MSE:0.000161 MAE: 0.009495 gini: 0.909457 R2: 0.900399 MAPE: 0.011533
    	[4] Test: MSE:0.000182 MAE: 0.010022 gini: 0.933888 R2: 0.887558 MAPE: 0.012314
    	[5] Test: MSE:0.000147 MAE: 0.008902 gini: 0.946892 R2: 0.909089 MAPE: 0.010856
    	[6] Test: MSE:0.000115 MAE: 0.007516 gini: 0.95848 R2: 0.929167 MAPE: 0.009215
    	[7] Test: MSE:0.000152 MAE: 0.009242 gini: 0.998578 R2: 0.906032 MAPE: 0.011301
    	[8] Test: MSE:0.000164 MAE: 0.009938 gini: 0.925367 R2: 0.898473 MAPE: 0.012055
    	[9] Test: MSE:0.000112 MAE: 0.007541 gini: 0.918505 R2: 0.930832 MAPE: 0.00921
    	[10] Test: MSE:0.000109 MAE: 0.007328 gini: 0.928719 R2: 0.932866 MAPE: 0.008957
    
    
    	[1] Test: MSE:0.000424 MAE: 0.017035 gini: 0.939391 R2: 0.738143 MAPE: 0.020684
    	[2] Test: MSE:0.000166 MAE: 0.009495 gini: 0.881094 R2: 0.897251 MAPE: 0.011547
    	[3] Test: MSE:0.000164 MAE: 0.009542 gini: 0.887316 R2: 0.898718 MAPE: 0.011584
    	[4] Test: MSE:0.000127 MAE: 0.007968 gini: 0.961209 R2: 0.921612 MAPE: 0.009771
    	[5] Test: MSE:0.000156 MAE: 0.009121 gini: 0.897298 R2: 0.903945 MAPE: 0.011263
    	[6] Test: MSE:0.000118 MAE: 0.007692 gini: 0.933346 R2: 0.927043 MAPE: 0.009455
    	[7] Test: MSE:0.000182 MAE: 0.01033 gini: 0.936644 R2: 0.88761 MAPE: 0.0127
    	[8] Test: MSE:0.000106 MAE: 0.007144 gini: 0.925808 R2: 0.934805 MAPE: 0.008759
    	[9] Test: MSE:0.00012 MAE: 0.00797 gini: 0.935092 R2: 0.926016 MAPE: 0.009718
    	[10] Test: MSE:0.000103 MAE: 0.007038 gini: 0.96785 R2: 0.936545 MAPE: 0.008631
    
    
    	[1] Test: MSE:0.000191 MAE: 0.010043 gini: 0.930669 R2: 0.881849 MAPE: 0.012285
    	[2] Test: MSE:0.000239 MAE: 0.012081 gini: 0.912542 R2: 0.85227 MAPE: 0.014648
    	[3] Test: MSE:0.000153 MAE: 0.008996 gini: 0.916455 R2: 0.905708 MAPE: 0.010968
    	[4] Test: MSE:0.000139 MAE: 0.008431 gini: 0.902012 R2: 0.914255 MAPE: 0.010304
    	[5] Test: MSE:0.000135 MAE: 0.008451 gini: 0.911053 R2: 0.916404 MAPE: 0.010294
    	[6] Test: MSE:0.000129 MAE: 0.008114 gini: 0.939414 R2: 0.920345 MAPE: 0.009977
    	[7] Test: MSE:0.000231 MAE: 0.012297 gini: 0.957719 R2: 0.857281 MAPE: 0.015055
    	[8] Test: MSE:0.000113 MAE: 0.00748 gini: 0.993521 R2: 0.930126 MAPE: 0.009168
    	[9] Test: MSE:0.000122 MAE: 0.008006 gini: 0.894544 R2: 0.924937 MAPE: 0.009735
    	[10] Test: MSE:0.000126 MAE: 0.008209 gini: 0.902523 R2: 0.922084 MAPE: 0.009974
    
    
    	[1] Test: MSE:0.000202 MAE: 0.010387 gini: 0.918051 R2: 0.875259 MAPE: 0.012679
    	[2] Test: MSE:0.00015 MAE: 0.008777 gini: 0.907705 R2: 0.907646 MAPE: 0.010736
    	[3] Test: MSE:0.000153 MAE: 0.009069 gini: 0.935907 R2: 0.905336 MAPE: 0.011057
    	[4] Test: MSE:0.000203 MAE: 0.011176 gini: 0.950277 R2: 0.874687 MAPE: 0.013593
    	[5] Test: MSE:0.000162 MAE: 0.00959 gini: 0.944633 R2: 0.899964 MAPE: 0.011673
    	[6] Test: MSE:0.000132 MAE: 0.008294 gini: 0.953197 R2: 0.918346 MAPE: 0.010186
    	[7] Test: MSE:0.000142 MAE: 0.00872 gini: 0.932828 R2: 0.912403 MAPE: 0.010734
    	[8] Test: MSE:0.000122 MAE: 0.00792 gini: 0.943742 R2: 0.924461 MAPE: 0.009741
    	[9] Test: MSE:0.000108 MAE: 0.007291 gini: 0.902824 R2: 0.933326 MAPE: 0.008921
    	[10] Test: MSE:0.000106 MAE: 0.007158 gini: 0.912962 R2: 0.934381 MAPE: 0.008805
    
    
    MAPE: 0.010769100501334554


## **Original output from [REPO](https://github.com/entron/entity-embedding-rossmann) code**:
    
`
Using TensorFlow backend.
Number of samples used for training: 200000
Fitting NN_with_EntityEmbedding...
Train on 200000 samples, validate on 84434 samples
Epoch 1/10
200000/200000 [==============================] - 13s 64us/step - loss: 0.0142 - val_loss: 0.0119
Epoch 2/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0096 - val_loss: 0.0109
Epoch 3/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0089 - val_loss: 0.0113
Epoch 4/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0082 - val_loss: 0.0101
Epoch 5/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0077 - val_loss: 0.0101
Epoch 6/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0074 - val_loss: 0.0100
Epoch 7/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0072 - val_loss: 0.0099
Epoch 8/10
200000/200000 [==============================] - 12s 59us/step - loss: 0.0071 - val_loss: 0.0096
Epoch 9/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0069 - val_loss: 0.0092
Epoch 10/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0068 - val_loss: 0.0095
Result on validation data:  0.10152226095724903
Train on 200000 samples, validate on 84434 samples
Epoch 1/10
200000/200000 [==============================] - 13s 63us/step - loss: 0.0140 - val_loss: 0.0117
Epoch 2/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0093 - val_loss: 0.0107
Epoch 3/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0084 - val_loss: 0.0109
Epoch 4/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0079 - val_loss: 0.0096
Epoch 5/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0076 - val_loss: 0.0097
Epoch 6/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0074 - val_loss: 0.0097
Epoch 7/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0072 - val_loss: 0.0097
Epoch 8/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0070 - val_loss: 0.0093
Epoch 9/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0069 - val_loss: 0.0094
Epoch 10/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0068 - val_loss: 0.0093
Result on validation data:  0.10194501967184522
Train on 200000 samples, validate on 84434 samples
Epoch 1/10
200000/200000 [==============================] - 13s 64us/step - loss: 0.0141 - val_loss: 0.0121
Epoch 2/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0093 - val_loss: 0.0100
Epoch 3/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0084 - val_loss: 0.0098
Epoch 4/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0079 - val_loss: 0.0095
Epoch 5/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0076 - val_loss: 0.0098
Epoch 6/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0074 - val_loss: 0.0097
Epoch 7/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0072 - val_loss: 0.0098
Epoch 8/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0071 - val_loss: 0.0092
Epoch 9/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0070 - val_loss: 0.0093
Epoch 10/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0069 - val_loss: 0.0097
Result on validation data:  0.10076855799458961
Train on 200000 samples, validate on 84434 samples
Epoch 1/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0141 - val_loss: 0.0114
Epoch 2/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0093 - val_loss: 0.0105
Epoch 3/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0084 - val_loss: 0.0108
Epoch 4/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0079 - val_loss: 0.0099
Epoch 5/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0076 - val_loss: 0.0098
Epoch 6/10
200000/200000 [==============================] - 12s 60us/step - loss: 0.0074 - val_loss: 0.0099
Epoch 7/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0072 - val_loss: 0.0100
Epoch 8/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0071 - val_loss: 0.0095
Epoch 9/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0070 - val_loss: 0.0096
Epoch 10/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0068 - val_loss: 0.0099
Result on validation data:  0.10973501886112967
Train on 200000 samples, validate on 84434 samples
Epoch 1/10
200000/200000 [==============================] - 13s 63us/step - loss: 0.0144 - val_loss: 0.0116
Epoch 2/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0094 - val_loss: 0.0109
Epoch 3/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0084 - val_loss: 0.0103
Epoch 4/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0079 - val_loss: 0.0099
Epoch 5/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0076 - val_loss: 0.0104
Epoch 6/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0074 - val_loss: 0.0099
Epoch 7/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0072 - val_loss: 0.0099
Epoch 8/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0070 - val_loss: 0.0099
Epoch 9/10
200000/200000 [==============================] - 12s 62us/step - loss: 0.0069 - val_loss: 0.0097
Epoch 10/10
200000/200000 [==============================] - 12s 61us/step - loss: 0.0068 - val_loss: 0.0100
Result on validation data:  0.10491748954856149
`

## **XGB performance**:


```python
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
print('MAPE: %s' % eval_utils.MAPE(
    y_true=y_test, 
    y_pred=xgb_test_y_pred))
```

    MAPE: 0.14712066617861289


## **Original output from [REPO](https://github.com/entron/entity-embedding-rossmann) code**:

`
.
.
.
[2987]  train-rmse:0.148847
[2988]  train-rmse:0.148845
[2989]  train-rmse:0.148842
[2990]  train-rmse:0.148839
[2991]  train-rmse:0.148834
[2992]  train-rmse:0.148819
[2993]  train-rmse:0.148768
[2994]  train-rmse:0.148762
[2995]  train-rmse:0.148741
[2996]  train-rmse:0.148705
[2997]  train-rmse:0.148667
[2998]  train-rmse:0.148622
[2999]  train-rmse:0.148584
Result on validation data:  0.14691216270195093
`

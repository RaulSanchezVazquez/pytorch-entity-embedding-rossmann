
# Entity-embedding-rossmann 

This is a Pytorch implementation with sklearn model interface.

This reproduces the code used in the paper **"[Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737)"**. 

Original version in Keras version can be found in: 
**Entity Embeddings of Categorical Variables [REPO](https://github.com/entron/entity-embedding-rossmann)**.



```python
import pandas as pd
import datasets
import eval_utils
import numpy as np

import EENN as eenn

X, y, X_test, y_test = datasets.get_X_train_test_data()

for data in [X, X_test]:
    data.drop('Open', inplace=True, axis=1)

models = []
for _ in range(5):
    m = eenn.EntEmbNNRegression(
        cat_emb_dim={
            'Store': 10,
            'DayOfWeek': 6,
            'Promo': 1,
            'Year': 2,
            'Month': 6,
            'Day': 10,
            'State': 6},
        alpha=0,
        dense_layers = [1000, 500],
        drop_out_layers = [0., 0.],
        drop_out_emb = 0.,
        loss_function='L1Loss',
        train_size = 1.0,
        y_max = max(y.max(), y_test.max())
        )

    m.fit(X, y)
    models.append(m)
    print('\n')

test_y_pred = np.array([model.predict(X_test) for model in models])
test_y_pred = test_y_pred.mean(axis=0)

print('MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))
```

    	[1] Test: MSE:0.00018 MAE: 0.00939 gini: 0.90037 R2: 0.89092 MAPE: 0.01153
    	[2] Test: MSE:0.00014 MAE: 0.00825 gini: 0.92758 R2: 0.91568 MAPE: 0.01012
    	[3] Test: MSE:0.00013 MAE: 0.00786 gini: 0.93557 R2: 0.92246 MAPE: 0.00966
    	[4] Test: MSE:0.00011 MAE: 0.00729 gini: 0.93948 R2: 0.93083 MAPE: 0.00895
    	[5] Test: MSE:0.00011 MAE: 0.00734 gini: 0.94207 R2: 0.93115 MAPE: 0.00902
    	[6] Test: MSE:0.00011 MAE: 0.00709 gini: 0.94487 R2: 0.93505 MAPE: 0.00871
    	[7] Test: MSE:0.00011 MAE: 0.00714 gini: 0.93532 R2: 0.93431 MAPE: 0.00879
    	[8] Test: MSE:0.0001 MAE: 0.00683 gini: 0.94437 R2: 0.93877 MAPE: 0.00839
    	[9] Test: MSE:0.0001 MAE: 0.00663 gini: 0.95925 R2: 0.94113 MAPE: 0.00814
    	[10] Test: MSE:9e-05 MAE: 0.00654 gini: 0.9411 R2: 0.94244 MAPE: 0.00802
    
    
    	[1] Test: MSE:0.00018 MAE: 0.00954 gini: 0.90326 R2: 0.88809 MAPE: 0.01171
    	[2] Test: MSE:0.00015 MAE: 0.00866 gini: 0.9222 R2: 0.90806 MAPE: 0.01064
    	[3] Test: MSE:0.00013 MAE: 0.00801 gini: 0.93858 R2: 0.91997 MAPE: 0.00985
    	[4] Test: MSE:0.00011 MAE: 0.00732 gini: 0.93974 R2: 0.93037 MAPE: 0.00897
    	[5] Test: MSE:0.00014 MAE: 0.0086 gini: 0.95085 R2: 0.91388 MAPE: 0.01056
    	[6] Test: MSE:0.00011 MAE: 0.00718 gini: 0.94448 R2: 0.93322 MAPE: 0.00882
    	[7] Test: MSE:0.00011 MAE: 0.00705 gini: 0.9483 R2: 0.93516 MAPE: 0.00867
    	[8] Test: MSE:0.0001 MAE: 0.00688 gini: 0.95069 R2: 0.93765 MAPE: 0.00845
    	[9] Test: MSE:0.0001 MAE: 0.00673 gini: 0.92546 R2: 0.93959 MAPE: 0.00829
    	[10] Test: MSE:9e-05 MAE: 0.00655 gini: 0.94907 R2: 0.94207 MAPE: 0.00805
    
    
    	[1] Test: MSE:0.00018 MAE: 0.00953 gini: 0.90303 R2: 0.88811 MAPE: 0.01167
    	[2] Test: MSE:0.00015 MAE: 0.00839 gini: 0.92067 R2: 0.90917 MAPE: 0.01032
    	[3] Test: MSE:0.00013 MAE: 0.00775 gini: 0.93699 R2: 0.92255 MAPE: 0.00953
    	[4] Test: MSE:0.00013 MAE: 0.00793 gini: 0.91284 R2: 0.92119 MAPE: 0.00977
    	[5] Test: MSE:0.00011 MAE: 0.00736 gini: 0.92173 R2: 0.92963 MAPE: 0.00906
    	[6] Test: MSE:0.00011 MAE: 0.00707 gini: 0.9244 R2: 0.93424 MAPE: 0.00869
    	[7] Test: MSE:0.00011 MAE: 0.00716 gini: 0.92702 R2: 0.93429 MAPE: 0.00881
    	[8] Test: MSE:0.0001 MAE: 0.0067 gini: 0.95261 R2: 0.94005 MAPE: 0.00823
    	[9] Test: MSE:9e-05 MAE: 0.00653 gini: 0.9407 R2: 0.94226 MAPE: 0.00802
    	[10] Test: MSE:9e-05 MAE: 0.00655 gini: 0.94767 R2: 0.94267 MAPE: 0.00802
    
    
    	[1] Test: MSE:0.00018 MAE: 0.00971 gini: 0.90033 R2: 0.88663 MAPE: 0.01194
    	[2] Test: MSE:0.00014 MAE: 0.00847 gini: 0.93348 R2: 0.9108 MAPE: 0.01042
    	[3] Test: MSE:0.00012 MAE: 0.0076 gini: 0.93482 R2: 0.9261 MAPE: 0.00932
    	[4] Test: MSE:0.00011 MAE: 0.00745 gini: 0.9374 R2: 0.9291 MAPE: 0.00916
    	[5] Test: MSE:0.00013 MAE: 0.008 gini: 0.92729 R2: 0.92249 MAPE: 0.00986
    	[6] Test: MSE:0.0001 MAE: 0.00691 gini: 0.94084 R2: 0.93694 MAPE: 0.00849
    	[7] Test: MSE:0.0001 MAE: 0.00687 gini: 0.94664 R2: 0.93756 MAPE: 0.00844
    	[8] Test: MSE:0.0001 MAE: 0.00688 gini: 0.94617 R2: 0.93778 MAPE: 0.00846
    	[9] Test: MSE:9e-05 MAE: 0.00658 gini: 0.93554 R2: 0.94156 MAPE: 0.00809
    	[10] Test: MSE:9e-05 MAE: 0.0065 gini: 0.96267 R2: 0.9426 MAPE: 0.00797
    
    
    	[1] Test: MSE:0.0002 MAE: 0.01033 gini: 0.90943 R2: 0.87382 MAPE: 0.01271
    	[2] Test: MSE:0.00015 MAE: 0.00846 gini: 0.92154 R2: 0.91024 MAPE: 0.01038
    	[3] Test: MSE:0.00012 MAE: 0.00773 gini: 0.90986 R2: 0.92311 MAPE: 0.0095
    	[4] Test: MSE:0.00011 MAE: 0.0073 gini: 0.92244 R2: 0.93035 MAPE: 0.00897
    	[5] Test: MSE:0.00012 MAE: 0.00768 gini: 0.92714 R2: 0.92641 MAPE: 0.00946
    	[6] Test: MSE:0.00011 MAE: 0.00712 gini: 0.94026 R2: 0.93408 MAPE: 0.00875
    	[7] Test: MSE:0.0001 MAE: 0.00688 gini: 0.93737 R2: 0.9375 MAPE: 0.00846
    	[8] Test: MSE:0.0001 MAE: 0.00669 gini: 0.9194 R2: 0.94 MAPE: 0.00823
    	[9] Test: MSE:0.0001 MAE: 0.00694 gini: 0.92574 R2: 0.93676 MAPE: 0.00856
    	[10] Test: MSE:9e-05 MAE: 0.00659 gini: 0.93111 R2: 0.94147 MAPE: 0.00811
    
    
    MAPE: 0.09784227344171176


## **Original output from [REPO](https://github.com/entron/entity-embedding-rossmann) code**:
    
```
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
```

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

    MAPE: 0.14955005536869034


## **Original output from [REPO](https://github.com/entron/entity-embedding-rossmann) code**:

```
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

```

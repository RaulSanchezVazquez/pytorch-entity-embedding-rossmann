
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

import datasets
import EntEmbNN as eenn

X_train, y_train, X_test, y_test = datasets.get_X_train_test_data()

for data in [X_train, X_test]:
    data.drop('Open', inplace=True, axis=1)

models = []
for _ in range(5):
    m = eenn.EntEmbNNRegression(
        X_y_test = (X_test, y_test),
        cat_emb_dim={
            'Store': 10,
            'DayOfWeek': 6,
            'Promo': 1,
            'Year': 2,
            'Month': 6,
            'Day': 10,
            'State': 6})
    
    m.fit(X_train, y_train)
    models.append(m)
    print('\n')

test_y_pred = np.array([model.predict(X_test) for model in models])
test_y_pred = test_y_pred.mean(axis=0)

print('MAPE: %s' % eval_utils.MAPE(
    y_true=y_test.values.flatten(),
    y_pred=test_y_pred))
```

    /home/raul/embeddings/EntEmbNN.py:397: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
      self.log_losses.append(loss.data[0])


    	[1] loss_train:0.00951 loss_test:0.01147
    	[2] loss_train:0.00884 loss_test:0.01089
    	[3] loss_train:0.00775 loss_test:0.01016
    	[4] loss_train:0.00729 loss_test:0.01016
    	[5] loss_train:0.00737 loss_test:0.00996
    	[6] loss_train:0.00693 loss_test:0.00966
    	[7] loss_train:0.00754 loss_test:0.01088
    	[8] loss_train:0.0065 loss_test:0.0102
    	[9] loss_train:0.00675 loss_test:0.00948
    	[10] loss_train:0.00654 loss_test:0.00964
    
    
    	[1] loss_train:0.00938 loss_test:0.01126
    	[2] loss_train:0.00853 loss_test:0.01071
    	[3] loss_train:0.00836 loss_test:0.01147
    	[4] loss_train:0.00754 loss_test:0.01061
    	[5] loss_train:0.00728 loss_test:0.00996
    	[6] loss_train:0.00742 loss_test:0.0099
    	[7] loss_train:0.00662 loss_test:0.00962
    	[8] loss_train:0.00662 loss_test:0.00992
    	[9] loss_train:0.00662 loss_test:0.00956
    	[10] loss_train:0.00688 loss_test:0.01009
    
    
    	[1] loss_train:0.00964 loss_test:0.01127
    	[2] loss_train:0.00957 loss_test:0.01116
    	[3] loss_train:0.0086 loss_test:0.01167
    	[4] loss_train:0.00742 loss_test:0.01032
    	[5] loss_train:0.00724 loss_test:0.01058
    	[6] loss_train:0.00709 loss_test:0.00985
    	[7] loss_train:0.00688 loss_test:0.01003
    	[8] loss_train:0.00666 loss_test:0.01054
    	[9] loss_train:0.00649 loss_test:0.01036
    	[10] loss_train:0.00663 loss_test:0.01083
    
    
    	[1] loss_train:0.00982 loss_test:0.01154
    	[2] loss_train:0.00886 loss_test:0.01025
    	[3] loss_train:0.00782 loss_test:0.01089
    	[4] loss_train:0.00726 loss_test:0.00991
    	[5] loss_train:0.00738 loss_test:0.01071
    	[6] loss_train:0.00839 loss_test:0.01166
    	[7] loss_train:0.00707 loss_test:0.00955
    	[8] loss_train:0.00666 loss_test:0.00992
    	[9] loss_train:0.0065 loss_test:0.00924
    	[10] loss_train:0.00669 loss_test:0.00995
    
    
    	[1] loss_train:0.01091 loss_test:0.01317
    	[2] loss_train:0.00833 loss_test:0.0107
    	[3] loss_train:0.00771 loss_test:0.01018
    	[4] loss_train:0.00752 loss_test:0.01016
    	[5] loss_train:0.00732 loss_test:0.00999
    	[6] loss_train:0.00687 loss_test:0.00976
    	[7] loss_train:0.00675 loss_test:0.00987
    	[8] loss_train:0.00655 loss_test:0.00937
    	[9] loss_train:0.00653 loss_test:0.00942
    	[10] loss_train:0.00729 loss_test:0.00966
    
    
    MAPE: 0.09726746216202672


**Original output from "[REPO](https://github.com/entron/entity-embedding-rossmann)" code**:
    
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


**Original output from [REPO](https://github.com/entron/entity-embedding-rossmann) code**:


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

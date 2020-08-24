## TCN Transactions Fraud Score for Customer Risk Model

This directory contains the application of `Temporal Convolutional Networks (TCN)` to the SoFi Money transactions data to predict a fraud score, then treat the fraud score as an input feature toward the `Customer Risk Lightgbm` model. More details can be found in this [deck](https://github.com/wangzhongruo/customer_risk_model/blob/master/Risk%20model%20-%20TCN%20Presentation.pdf).

**This version is currently in production.**

### How to use TCN binary classifier
--- 
#### Select Features
---

First select the relevent columns used for training. 
* `target_col`: column name of the target variable.
* `meta_cols`: column names of the meta data needed in order to identify the predictions.
    * to match prediction and datapoint, join resulting prediction dataframe based on meta_cols.
* `features`: features used for TCN.

```
# select columns
target_col=['target']
meta_cols=['borrower_id', 'index']
features = ['nr_past_transactions', 'card_present_ind', 'has_transaction_comment', 
            'hr_of_transaction', 'international_transaction_flag', 'transaction_code',
            'transaction_type', 'days_since_first_transaction', 'transaction_as_pct_of_balance']


df_train = df_train_trans[features + target_col + meta_cols].copy()
df_test = df_test_trans[features + target_col + meta_cols].copy()
```

#### Train
---

How to train the TCN model.

```
# feature embedding dimensions for categorical features
feature_embed_dims = {'transaction_code': 20,
                      'transaction_type': 10} 

# weights for the positive class
pos_weight = len(df_train) / df_train.target.value_counts().loc[True]

# insert a model directory to store the computation and logs
model_dir = "../../model/TCN_CLF_TRANS_dev/"
tcn = TCNs.TCNClassifier(tcn_layers="200 200 200", 
                         mlp_layers="200 200 100", 
                         epoch=10, 
                         kernel_size=5,
                         batch_size=32, 
                         device="cuda:0", 
                         dropout_mlp=0.5, 
                         dropout_tcn=0.2, 
                         feature_embed_dims=feature_embed_dims, 
                         patience=10, 
                         model_dir=model_dir,
                         lr=0.0002, 
                         print_freq=300,
                         lr_decay_freq=3, 
                         weight_decay=1e-6,
                         log_path=os.path.join(model_dir, 'log.txt'), 
                         verbose=1,
                         min_length=1)

# for now, group_by only takes one item
print("\nprocessing data")

# process train and test dataframes using tcn.process_data to get them into the right format
data = tcn.process_data(df_train, 'target', df_test, sort_by_col=['index'], group_by=['borrower_id'])
test_data = tcn.process_data_inference(df_test, target_col='target', sort_by_col=['index'], group_by=['borrower_id'])

# reset/update TCN parameters by calling:
tcn.set_params(epoch=5, lr=0.0001)  # set epochs to 5 and learning rate to 0.0001

# call tcn.fit on processed data object to fit the model
tcn.fit(data)

```

#### Inference
---

It's preferred to load back the previously computed model, instead of evaluting `tcn` directly due to the use of early stopping. The stored model is less likely to be overfitted than the current one. 

```
# load back the model saved at the early stopping iteration, to cpu
tcn = torch.load("../model/TCN_CLF_TRANS_dev/tcn.pth", map_location='cpu')

# process inference dataframe
test_data = tcn.process_data_inference(df_test, target_col='target', sort_by_col=['index'], group_by=['borrower_id'])

# make predictions. 
# given a dataframe with multiple sequences, return predictions at all timesteps. 
preds = tcn.predict_proba(test_data, return_sequences=False)    

# given a dataframe with multiple sequences, return predictions at the last timestep.
preds_seq = tcn.predict_proba(test_data, return_sequences=True)

# join the preds dataframe with df_test based on the 'index' column
inference_pred = pd.merge(preds, df_test, on=['index'], how='inner')

# print the roc_auc score of the model prediction
print(roc_auc_score(y_true=inference_pred.target, y_score=inference_pred.tcn_pred))
inference_pred.head()
```


#### Inference a single user.
---

Given a borrower's latest k transactions, predict the risk score at the last time step. This means for a sequence of inputs, the result is a dataframe with a `borrow_id` along with it's prediction `tcn_pred`.

1. make sure `df_test` has the following columns
```
features = ['nr_past_transactions', 'card_present_ind', 'has_transaction_comment', 
            'hr_of_transaction', 'international_transaction_flag', 'transaction_code',
            'transaction_type', 'days_since_first_transaction', 'transaction_as_pct_of_balance']
```
2. include a `borrower_id` column as a placeholder. It can be anything as long as it has an one-to-one relationship with users.

3. Run the following code:
```
test_data = tcn.process_data_inference(df_test, sort_by_col=[], group_by=["borrower_id"])
tcn.set_params(min_length=1, device='cpu')
preds = tcn.predict_proba(test_data, return_sequences=False)  
tcn_pred = float(preds.tcn_pred.iloc[0])
```

4. in the above snippet, `preds` is a dataframe containing the `borrower_id` and predicted score `tcn_pred`. You can retrieve the raw score by calling `tcn_pred = float(preds.tcn_pred.iloc[0])`.


### Directory:
---
#### Notebooks:
* `customer_risk_model.ipynb`: all code to the training stage of TCN component of the customer risk model
* `customer_risk_model_validation.ipynb`: all code to the validation stage of TCN component.
* `customer_risk_model_inference.ipynb`: after training a few TCN on the entire dataset, use one to inference for the production model lgbm model.

#### Source code:
* `TCNs.py` contains the high-level API of the TCN binary classifier used in this task.
* If you would like to use this repo as a starting point for your customized model, the following files will be more important. 
    * `dataloader.py`
    * `model.py`
    * `train.py`
    * `utils.py`
    * `metrics.py`
    * `tcn.py`
* `main.py` is used to train models with commandline, so you can train multiple models sequentially and analyze it later.
    
**TODO: more detailed description on the files**

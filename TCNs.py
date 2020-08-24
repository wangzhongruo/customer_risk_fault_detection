import importlib
import dataloader
import train
import model as tcn_model
import utils

import dill
import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import random
import torch



def df_to_list(df, sort_by=[], group_by=[], return_index=False, min_length=1):
    
    def fn(df_, data_):
        if len(group_by) == 0:
            data_['cur_user'] = df_.values
            return None
        
        
        uid = df_[group_by].iloc[0][group_by[0]] # extract the identifier from group_by
        if len(sort_by) > 0:
            df_tmp = df_.sort_values(sort_by)
        else:
            df_tmp = df_
        
        d = df_tmp.values
        
        # abandon the datum if it's length < min_length
        if len(d) < min_length:
            return None
        
        # add datum to the dictionary of data
        if return_index:
            i = df_tmp.index.to_list()
            data_[uid] = (d, i)
        else:
            data_[uid] = d
        return None

    data = {}
    if len(group_by) > 0:
        df.groupby(group_by).apply(lambda x: fn(x, data))
    else:
        df.apply(lambda x: fn(x, data))
        
    if return_index:
        data, index = list(zip(*data.values()))
        return data, index
    
    data = list(data.values())
    return data
    
    
    
def get_features(df, target=[], meta=[]):
    """
    1. obtain the features
    2. set order for the features
    3. reindex the df
    """
    ############################################################
    #                   Type conversion
    ############################################################

    types = df[df.columns[~df.columns.isin(target+meta)]].dtypes
    for col_name, col_type in types.iteritems():
        if col_type == bool:
            df[col_name] = df[col_name].astype(float)

    ############################################################
    #                 Get features by type
    ############################################################
    
    features_cat = filter(lambda x: not np.issubdtype(x[1], np.number), types.iteritems())
    features_cat = sorted(list(map(lambda x: x[0], features_cat)))
    # target and meta should have already been removed. but just to be sure
    features_num = sorted(list(set(types.index) - set(features_cat) - set(target) - set(meta))) 
    selected_features = df.columns.to_list()
    features_idx = dict(zip(selected_features, range(len(selected_features))))
    
    return selected_features, features_num, features_cat, features_idx


def groupby_train_test_split(df, selected_features=None, test_ratio=0.2, seed=12345, groupby='user_id'):
    """
    this takes in the train and test data combined as a single df,
    convert columns to appropriate types, and return train-test-splitted data
    """

    ############################################################
    #                   Train Test Split
    ############################################################

    grp = df[groupby]
    n_splits = int(1 / test_ratio)
    groupkfold = GroupKFold(n_splits=n_splits)
    random.seed(seed)
    folds = groupkfold.split(df, groups = grp)
    train_idx, test_idx = next(folds)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    
    return df_train, df_test
    

        
class TCNClassifier:
    def __init__(self, tcn_layers, mlp_layers, epoch, kernel_size, 
                 batch_size=32, data_fraction=1, device='cpu', dropout_mlp=1, dropout_tcn=1, 
                 lr=0.0005, lr_decay_freq=5, model_dir=None, output_size=1, pad_after=True, patience=5, 
                 pos_weight=1, print_freq=200, seed=12345, weight_decay=0, default_embed_dim=10, vocabs=None, 
                 verbose=1, feature_embed_dims=None, log_path=None, warm_start=True, min_length=1):
        self.batch_size = batch_size
        self.data_fraction = data_fraction
        self.device = device
        self.dropout_tcn = dropout_tcn
        self.dropout_mlp = dropout_mlp
        self.epoch = epoch
        self.mlp_layers = utils.parse_ints(mlp_layers)
        self.kernel_size = kernel_size
        self.tcn_layers = utils.parse_ints(tcn_layers)
        self.lr = lr
        self.lr_decay_freq = lr_decay_freq
        self.model_dir = model_dir
        self.pad_after = pad_after 
        self.patience = patience
        self.print_freq = print_freq 
        self.weight_decay = weight_decay
        self.output_size = output_size
        self.pos_weight = pos_weight
        self.seed = seed
        self.default_embed_dim = default_embed_dim
        self.vocabs = vocabs
        self.feature_embed_dims = feature_embed_dims
        self.log_path = log_path
        self.verbose = verbose
        self.warm_start = warm_start
        self.min_length = min_length
        
        assert(self.mlp_layers[0] == self.tcn_layers[-1])
            
    
    def process_data(self, X_train, target_col, X_valid=None, y_valid=None, sort_by_col=[], group_by=[], meta=None):
        """
        There are two ways to process this dataset:
        

        X is a list of pd.DataFrames, and y is a column name in X.
           type(y) == str
        """
        data = {}    
        target = [target_col]
        meta = sort_by_col + group_by
        selected_features, features_num, features_cat, features_idx = get_features(X_train, target, meta)
        
        train_data = df_to_list(X_train, sort_by=sort_by_col, group_by=group_by, return_index=False, min_length=self.min_length)
        data['train'] = train_data
        if X_valid is not None:
            test_data = df_to_list(X_valid, sort_by=sort_by_col, group_by=group_by, return_index=False, min_length=self.min_length)
            data['test'] = test_data
        
        ############################################################
        #                   build data object
        ############################################################

        # index
        data['features'] = selected_features
        data['features_num'] = features_num
        data['features_cat'] = features_cat
        data['source_features'] = features_num + features_cat
        data['features_target'] = target
        data['features_meta'] = meta
        data['features_idx'] = features_idx
            
        print("data processed!")
        return data

    
    def process_data_inference(self, X_train, sort_by_col=[], group_by=[], meta=None, target_col=None):
        """
        @param X_train: dataframe 
        @param sort_by_col: column to sort by within each groupby
        @param group_by: column to group by before split into data objects
        @param meta: list of column names of meta data
        @param target_col: name of the target column if it is within X_train. 
        """
        data = {}    
        meta = sort_by_col + group_by
        selected_features, features_num, features_cat, features_idx = get_features(X_train, [], meta)
        train_data, index = df_to_list(X_train, sort_by=sort_by_col, group_by=group_by, return_index=True, min_length=self.min_length)
        
        data['train'] = train_data
        data['index'] = index

        # index
        data['features'] = selected_features
        data['features_num'] = list(filter(lambda x: x not in meta + [target_col], features_num)) if target_col is not None else features_num
        data['features_cat'] = list(filter(lambda x: x not in meta + [target_col], features_cat)) if target_col is not None else features_cat
        data['features_meta'] = meta
        data['features_idx'] = features_idx
    
        print("data processed!")
        return data
    
    
    def process_feature_embed_dims(self, data):
        if self.feature_embed_dims is None:
            print("No feature embedding dimensions provided..., use default embedding dim: {}".format(self.default_embed_dim))
            self.features_embed_dims = {}
            for f in data['features_cat']:
                self.feature_embed_dims[f] = self.default_embed_dim
                
        elif set(data['features_cat']) != set(self.feature_embed_dims.keys()):
            print("unmatched embeding dimensions. Setting missing embed dims to default embedding dim")
            for f in data['features_cat']:
                if f not in self.feature_embed_dims:
                    print("setting embed dim to {} for feature: {}".format(self.default_embed_dim, f))
                    self.feature_embed_dims[f] = self.default_embed_dim
            
    
    def get_loader(self, data):
        loaders = dataloader.get_loader(copy.deepcopy(data), batch_size=self.batch_size, 
                                        pad_after=self.pad_after, vocabs=self.vocabs, inference=False)

        if len(loaders) == 2:
            train_loader, test_loader = loaders
        else:
            train_loader = loaders
            test_loader = None
            
        if self.verbose > 0:
            print("train loader length: {}".format(len(train_loader)))
            
        # set vocabs for later uses
        self.vocabs = {'vocab_lens': train_loader.dataset.vocab_lens,
                       'w2i_dicts': train_loader.dataset.w2i_dicts,
                       'i2w_dicts': train_loader.dataset.i2w_dicts}
        return train_loader, test_loader
    
        
    def set_vocabs(self, vocabs):
        self.vocabs = vocabs
        
        
    def get_inference_loader(self, data):
        if self.vocabs is None:
            print("No vocabs for categorical features provided...this is equivalent to randomly embed categorical features")
        train_loader = dataloader.get_loader(copy.deepcopy(data), batch_size=self.batch_size, 
                                             pad_after=self.pad_after, vocabs=self.vocabs, 
                                             inference=True)
        return train_loader
        
    
    def prep_model_params(self, data):
        assert(hasattr(self, 'vocabs'))
        
        self.params = {
            'source_features': data['source_features'],
            'features_num': data['features_num'],
            'features_cat': data['features_cat'],
            'embed_dims': self.feature_embed_dims,
            'vocab_lens': self.vocabs['vocab_lens'],
            'layers': self.tcn_layers, # make sure layers [-1] == in_layers[0]
            'in_layers': self.mlp_layers,
            'output_size': self.output_size, 
            'dropout_tcn': self.dropout_tcn,
            'dropout_mlp': self.dropout_mlp,
            'kernel_size': self.kernel_size,
            'device': self.device
        } 
        
        
       
    def fit(self, data, feature_embed_dims=None, warm_start=True):    
        random.seed(self.seed)
        train_loader, test_loader = self.get_loader(data)
        
        if warm_start and hasattr(self, 'model'): 
            self.model = train.train(self, self.model, train_loader, test_loader)
        elif hasattr(self, 'model'):
            print("\n a model already exists! \n")
            self.model = train.train(self, self.model, train_loader, test_loader)
        else:
            if feature_embed_dims is not None:
                self.feature_embed_dims = feature_embed_dims
                
            assert(hasattr(self, 'feature_embed_dims'))
            self.process_feature_embed_dims(data)

            self.prep_model_params(data)
            if self.verbose > 0:
                print("params: ", self.params)
            self.model = tcn_model.TCN_Deposit(self.params)

            print("saving vocabularies ... ")
            torch.save(self.vocabs, '{}/vocabs.pt'.format(self.model_dir))

            ##################################################################
            #                          Training
            ##################################################################

            # run training
            self.model = train.train(self, self.model, train_loader, test_loader)

        
    def predict(self, data):
        pass
    
    
    def predict_proba(self, data, return_sequences=False, return_df=True):
        """
        @param data: data object for inference
        @param return_sequences: whether to return outputs from all time steps (many-to-many prediction) shape = b x s
                                 or just the last one (many-to-one prediction) shape = b
        """
        loader = self.get_inference_loader(data)
        
        if not hasattr(self, 'model'):
            print("initializing new model...")
            if feature_embed_dims is not None:
                self.feature_embed_dims = feature_embed_dims
            self.process_feature_embed_dims(data)

            self.prep_model_params(data)
            self.model = tcn_model.TCN_Deposit(self.params)
        
        inference_formatter = train.inference(self, self.model, loader, return_sequences)
        if return_df:
            result = inference_formatter.get_inference(data['features_meta'], return_sequences)
        else:
            result = inference_formatter.get_inference(None, return_sequences)
        return result
        
    
    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k, v)
            
            
    def save(self, model_dir):        
        if hasattr(self, 'criterion'):
            self.criterion.reset_train()
            self.criterion.reset_test()
        if hasattr(self, 'train_metrics'):
            self.train_metrics.reset()
        if hasattr(self, 'test_metrics'):
            self.test_metrics.reset()
        if hasattr(self, 'valid_formatter'):
            self.valid_formatter.reset()
        if hasattr(self, 'valid_formatter_target'):
            self.valid_formatter_target.reset()
        if hasattr(self, 'inference_formatter'):
            self.inference_formatter.reset()
            
        path = os.path.join(model_dir, 'tcn.pth')
        torch.save(self, path)
        
        
            

# target=['target']
# meta=['user_id', 'transaction_datetime']
# selected_features, features_num, features_cat, features_idx = get_features(modeling_df, target=['target'], meta=['user_id', 'transaction_datetime'])
# df_train, df_test = groupby_train_test_split(modeling_df, groupby='user_id')

# feature_embed_dims = {'transaction_code': 20,
#                         'transaction_description': 20,
#                         'transaction_type': 10} 
# pos_weight = len(df_train) / df_train.target.value_counts().loc[True]
# TCN = TCNClassifier(tcn_layers='200 200 200 200 200', mlp_layers='200 100 50', epoch=1, kernel_size=4,
#                     batch_size=32, device='cuda:0', dropout_mlp=0.5, dropout_tcn=0.1, 
#                     model_dir='../model/TCN_CLF_TRANS_dev', feature_embed_dims=feature_embed_dims, 
#                     patience=10, log_path='../model/TCN_CLF_TRANS_dev/log.txt', verbose=1)

# fit both train and validation data
# data = TCN.process_data(X_train=df_train, target_col='target', X_valid=df_test, sort_by_col=['transaction_datetime'], group_by=['user_id'])
# TCN.fit(data)
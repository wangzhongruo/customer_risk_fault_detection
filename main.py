import argparse
import dill
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import random


from utils import get_args
import feather
import TCNs


def main():
    args = get_args()
    random.seed(args.seed)
    # read data
    print("\nloading data...")
    target=['target']
    meta=['borrower_id']
    
    # train on basic featues with both df_train and df_test
    if args.data_type == 'basic_features':
        features = ['nr_past_transactions',
                    'card_present_ind',
                    'has_transaction_comment',
                    'hr_of_transaction',
                    'international_transaction_flag',
                    'transaction_code',
                    'transaction_type',
                    'days_since_first_transaction',
                    'transaction_as_pct_of_balance',
                    'target',
                    'borrower_id',
                    'index']
        
        train_path = "../data/TCN_CLF_customer_risk_score/v2/df_train_trans.feather"
        load_path = "../data/TCN_CLF_customer_risk_score/v2/df_test_trans.feather"
        df_train = feather.read_dataframe(train_path)
        df_test = feather.read_dataframe(load_path)
        
        df_train = df_train[features]
        df_test = df_test[features]
        
    # train on all features with both df_train and df_test
    elif args.data_type == 'all_features':
        features = ['nr_past_returns', 'nr_past_transactions', 'nr_pos_transactions',
                    'nr_atm_transactions', 'nr_direct_deposits',
                    'rolling_trns_as_pct_of_bal', 'rolling_mean_acc_bal',
                    'nr_transactions_per_day', 'transaction_as_pct_of_bal_min',
                    'transaction_as_pct_of_balance', 'card_present_ind',
                    'has_transaction_comment', 'hr_of_transaction',
                    'international_transaction_flag', 'days_since_first_transaction',
                    'transaction_code', 'transaction_type',
                    'target', 'borrower_id', 'index']
        
        train_path = "../data/TCN_CLF_customer_risk_score/v2/df_train_trans.feather"
        load_path = "../data/TCN_CLF_customer_risk_score/v2/df_test_trans.feather"
        df_train = feather.read_dataframe(train_path)
        df_test = feather.read_dataframe(load_path)
        df_train = df_train[features]
        df_test = df_test[features]
        
    # train on all available data, no train-test-split
    elif args.data_type == 'all_data':
        features = ['nr_past_transactions',
                    'card_present_ind',
                    'has_transaction_comment',
                    'hr_of_transaction',
                    'international_transaction_flag',
                    'transaction_code',
                    'transaction_type',
                    'days_since_first_transaction',
                    'transaction_as_pct_of_balance',
                    'target',
                    'borrower_id',
                    'index']
        
        train_path = "../data/TCN_CLF_customer_risk_score/v2/modeling_df_trans.feather"
        df_train = feather.read_dataframe(train_path)
        df_train = df_train[features]
    
    
    ############################################################################
    #          remove borrowers with less than min_length transactions
    #          filter data by length of transactions
    ############################################################################
    
    modeling_df_trans = feather.read_dataframe("../data/TCN_CLF_customer_risk_score/v2/modeling_df_trans.feather")
    bids = modeling_df_trans.groupby('borrower_id')['borrower_id'].count()
    bids = bids[bids > args.min_length].index.values
    df_train = df_train[df_train.borrower_id.isin(bids)]
    if args.data_type != 'all_data':
        df_test = df_test[df_test.borrower_id.isin(bids)]
        
    #########################################################
    
    
    if args.verbose > 0:
        print("-" * 80)
        print("loading train data from: {}".format(train_path))
        print(df_train.info())
        print("model will be saved to {}\n".format(args.model_dir))

    feature_embed_dims = {'transaction_code': 20,
                          'transaction_type': 10} 
    
    pos_weight = len(df_train) / df_train.target.value_counts().loc[True]
    print("min_length: ", args.min_length)
    print("pos_weight: ", pos_weight)
    
    # init model
    print("-"*80)
    print("initializing model")
    tcn = TCNs.TCNClassifier(tcn_layers=args.tcn_layers, 
                             mlp_layers=args.mlp_layers, 
                             epoch=args.epoch, 
                             kernel_size=args.kernel_size,
                             batch_size=args.batch_size, 
                             device=args.device, 
                             dropout_mlp=args.dropout_mlp, 
                             dropout_tcn=args.dropout_tcn, 
                             model_dir=args.model_dir, 
                             feature_embed_dims=feature_embed_dims, 
                             patience=args.patience, 
                             lr=args.lr, 
                             print_freq=args.print_freq,
                             lr_decay_freq=args.lr_decay_freq, 
                             log_path=os.path.join(args.model_dir, 'log.txt'), 
                             verbose=args.verbose,
                             min_length=args.min_length)
  

    if args.data_type == 'basic_features':
        print("\nprocessing data")
        data = tcn.process_data(df_train, 'target', df_test, sort_by_col=['index'], group_by=['borrower_id'])

    elif args.data_type == 'all_features':
        print("\nprocessing data")
        data = tcn.process_data(df_train, 'target', df_test, sort_by_col=['index'], group_by=['borrower_id'])

    elif args.data_type == 'all_data':
        print("\nprocessing data")
        data = tcn.process_data(df_train, 'target', sort_by_col=['index'], group_by=['borrower_id'])


    print("\nfitting model")
    tcn.fit(data)
    tcn.save(model_dir=args.model_dir)
    
    print("\n training complete!")
    
if __name__ =='__main__':
    main()
    
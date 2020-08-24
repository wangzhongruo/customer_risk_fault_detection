import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util

from torch import nn, LongTensor
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import EarlyStopping, LossGenie, build_summary, sigmoid, log_status, Timer, InferenceFormatter, parse_pred, parse_meta
import metrics
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def predict(ep, model_object, model, datum):
    src, tgt, seq_len, meta = datum

    # construct input sequence for decoder
    # pad inputs_rev on the left with zeros to enforce teacher forcing
    # assume the src and tgt has shape batch_size x seq_len x feature_dim

    # move all features to device
    src = src.to(model_object.device)
    tgt = tgt.to(model_object.device)
    seq_len = seq_len.to(model_object.device)
    
    # make prediction
    out = model(src)  # normalized(out) = prob
    out = parse_pred(out, seq_len, return_sequences=True, min_length=model_object.min_length)
    tgt = parse_pred(tgt, seq_len, return_sequences=True, min_length=model_object.min_length).float() # bceloss
    meta = parse_meta(meta, seq_len, return_sequences=True, min_length=model_object.min_length)
    loss = model_object.criterion(out, tgt)
    prob = sigmoid(out.cpu().detach().numpy())
    
    return loss, prob, tgt, meta


def inference(model_object, model, loader, return_sequences=False):
    """
    @param model_object: object contains all variables. Can be any class
    @param model: a tcn model
    @param return_sequences: whether to return a sequence or the prediction at the last step
    """
    if model_object.device.lower() != 'cpu':
        model_object.device = 'cuda:0'
    model = model.to(model_object.device)
    model.eval()
    
    formatter = InferenceFormatter()
    for datum in loader:
        src, seq_len, meta = datum
        src = src.to(model_object.device)
        out = model(src)
        out = parse_pred(out, seq_len, return_sequences=return_sequences, min_length=model_object.min_length)
        meta = parse_meta(meta, seq_len, return_sequences=return_sequences, min_length=model_object.min_length)
        prob = sigmoid(out.cpu().detach().numpy())
        
        formatter.collect(prob, meta)

    return formatter
    


##################################################
#                train an epoch
##################################################


def train_epoch(ep, model_object, model, train_loader, test_loader=None):
    model.train()
    train_timer = Timer('M')
    model_object.criterion.train()
    model_object.train_metrics.reset()
    print_counter = 0  # counter for printing
     
    for datum in train_loader:

        model_object.optimizer.zero_grad()
        loss, prob, tgt, _ = predict(ep, model_object, model, datum)
        loss.backward()
        
        # grad normalization
        grad = clip_grad_norm_(model.parameters(), 10)
        model_object.train_metrics.collect(y_prob=prob, y_true=tgt)

        # update network
        model_object.optimizer.step()

        print_counter += 1
        if print_counter % model_object.print_freq == 0 and model_object.log_path is not None:
            train_auc = model_object.train_metrics.roc_auc()
            train_ap = model_object.train_metrics.average_precision()
            status = '[epoch {}], train batch {} - batch loss: {:.5f}, running train auc: {:.5f}, running train ap: {:.5f} - time taken: {:.2f} mins'
            status = status.format(ep, model_object.criterion.n_batch_train, model_object.criterion.get_running_loss(), 
                                   train_auc, train_ap, train_timer.time())
            log_status(model_object.log_path, status, init=False) if (model_object.log_path and model_object.verbose > 0) else None
         
            # evaluate on validation set
            if test_loader is not None:
                train_timer.pause()
                valid_epoch(ep, model_object, model, test_loader)
                train_timer.resume()
                model.train()
            
    # print and log status
    train_loss = model_object.criterion.get_running_loss()
    train_auc = model_object.train_metrics.roc_auc()
    train_acc = model_object.train_metrics.accuracy()
    train_ap = model_object.train_metrics.average_precision()
    
    if model_object.log_path:
        status = '[epoch {}], train loss: {:.5f}, train acc: {:.5f}, train auc: {:.5f}, train ap: {:.5f}, time taken: {:.2f} mins'
        status = status.format(ep, train_loss, train_acc, train_auc, train_ap, train_timer.time_since_start())
        log_status(model_object.log_path, status, init=False) if model_object.verbose > 0 else None
    
    # reset running criterion losses and timer
    model_object.criterion.reset_train()
    train_timer.reset()

    
##################################################
#                   validation
##################################################
    
    
def valid_epoch(ep, model_object, model, test_loader):
    model.eval()
    model_object.criterion.eval()

    valid_start = time.time()
    valid_timer = Timer('M')
    
    preds, targets = [], []
    model_object.valid_formatter = InferenceFormatter() # formatter for debugging purpose
    model_object.valid_formatter_target = InferenceFormatter() # formatter for debugging purpose
    model_object.test_metrics.reset()
    for datum in test_loader:
        loss, prob, tgt, meta = predict(ep, model_object, model, datum)
        model_object.test_metrics.collect(y_prob=prob, y_true=tgt)
        model_object.valid_formatter.collect(prob, meta) # formatter for debugging purpose
        model_object.valid_formatter_target.collect(tgt, meta) # formatter for debugging purpose
   
    valid_loss = model_object.criterion.get_running_loss()
    valid_auc = model_object.test_metrics.roc_auc()
    valid_acc = model_object.test_metrics.accuracy()
    valid_ap = model_object.test_metrics.average_precision()
    
    status = '[epoch {}], valid loss: {:.5f}, valid acc: {:.5f}, valid auc: {:.5f}, valid ap: {:0.5f}, time taken: {:.2f} mins'
    status = status.format(ep, valid_loss, valid_acc, valid_auc, valid_ap, valid_timer.time())
    
    if model_object.log_path:
        log_status(model_object.log_path, status, init=False) if model_object.verbose > 0 else None
        log_status(model_object.log_path, str(model_object.test_metrics.top_k_percentile(k=20, step=1).to_string()), init=False) if model_object.verbose > 1 else None

    model_object.early_stopper(model_object.criterion.get_last_valid_loss(), model, model_object)
    
    # reset validation losses and set criterion to training mode
    model_object.criterion.reset_test()
    model_object.criterion.train()

    

##################################################
#                    train
##################################################

    
def train(model_object, model, train_loader, test_loader=None):
    device = model_object.device
    if model_object.device.lower() != 'cpu':
        model_object.device = 'cuda:0'
        
    if model_object.log_path and model_object.verbose > 0:
        log_status(model_object.log_path, 'model_params: {}'.format(str(model_object.params)), init=False)
        
    model = model.to(model_object.device)
        
    model_object.optimizer = optim.Adam(model.parameters(), lr=model_object.lr, weight_decay=model_object.weight_decay)
    kwargs = {'pos_weight': torch.Tensor([model_object.pos_weight]).to(model_object.device), 
              'reduction': 'mean'}
    model_object.criterion = LossGenie('BCEWithLogitsLoss', **kwargs)
    
    # assisting modules
    model_object.lr_scheduler = optim.lr_scheduler.StepLR(model_object.optimizer, step_size=model_object.lr_decay_freq, gamma=0.5)
    model_object.early_stopper = EarlyStopping(model_object.model_dir, patience=model_object.patience, verbose=True)
    model_object.train_metrics = metrics.BinaryClfMetrics()
    model_object.test_metrics = metrics.BinaryClfMetrics()
    
    for ep in range(1, model_object.epoch+1):
        # train for an epoch
        train_epoch(ep, model_object, model, train_loader, test_loader)
        
        # validation for an epoch
        if test_loader is not None:
            valid_epoch(ep, model_object, model, test_loader)
        
        # return if early stop
        if model_object.early_stopper.early_stop:
            return model
        
        model_object.lr_scheduler.step()
        
        if model_object.log_path and model_object.verbose > 0:
            log_status(model_object.log_path, 'current lr: {}'.format(model_object.lr_scheduler.get_lr()), init=False)
            
    return model


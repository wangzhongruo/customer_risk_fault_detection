import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datetime
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser()

    ##################################################################
    #                       Parser
    ##################################################################
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    
    # model architecture
    parser.add_argument('--tcn_layers', type=str, default='500 500 500 500', 
                        help='seq of ints defines layer dimensions of TCN')
    parser.add_argument('--mlp_layers', type=str, default='500',
                        help='seq of ints defines layer dimensions of VAE MLP encoder')
    parser.add_argument('--dropout_mlp', type=float, default=0.5,
                        help='the good old dropout in (0, 1], for MLP')
    parser.add_argument('--dropout_tcn', type=float, default=0.2,
                        help='the good old dropout in (0, 1], for TCN')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size of the 1D convolusions')
    parser.add_argument('--pad_after', type=str, default='True',
                        help='Pad source sequence at the end if True else False')
    parser.add_argument('--ep_threshold', type=int, default=10,
                        help='which epoch to set weight to reconstruction loss to 1')
    parser.add_argument('--output_size', type=int, default=1, 
                        help='dim of the output layer of TCN')
    
    # debugging
    parser.add_argument('--cat_target', type=str, default='False')
    
    # embedding dims
    parser.add_argument('--transaction_code_embed', type=int, default=20)
    parser.add_argument('--transaction_type_embed', type=int, default=10)

    
    # training hyperparams and configurations
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)    
    parser.add_argument('--data_fraction', type=float, default=1)
    parser.add_argument('--patience', type=int, default=5)  
    parser.add_argument('--pos_weight', type=float, default=1)
    parser.add_argument('--print_freq', type=int, default=10000)
    parser.add_argument('--lr_decay_freq', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--verbose', type=int, default=1)

    
    # Data, model, and output directories
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--dataset', type=str)
    
    
    # Args needed to train with SageMaker
    parser.add_argument('--data_type', type=str, default='basic', 
                        help='to use "basic" or "all" features')
    parser.add_argument('--train_instance_type', type=str, default='ml.p3.2xlarge',
                        help='instance size to train on')
    parser.add_argument('--default_SM_dirs', type=str, default='false',
                        help='whether to use SageMaker default directories. Set to True when deploying')


    args, _ = parser.parse_known_args()
    return args
    

     
def get_time_str():
    now = datetime.datetime.now()
    tag = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    return "-".join(map(str, tag))



def parse_ints(string):
    """ 
    parse str of a seq of ints
    123 234 512 -> [123, 234, 512]
    """
    L = string.split(" ")
    L = filter(lambda x: len(x) > 0, L)
    return list(map(int, L))  


########################################     
#          load models 
########################################


# def load_model(ckpt_path, all_params_path, vocab_path=None):
#     """
#     load low-level 
#     """
#     checkpoint = torch.load(ckpt_path, lambda storage, loc: storage)
#     all_params = torch.load(all_params_path, lambda storage, loc: storage)
#     all_params['params']['cat_target']=False
    
#     model = TCN_Deposit(all_params['params'])
#     model.load_state_dict(checkpoint['model'])
# #     model.train_losses = checkpoint['train_losses']
# #     model.train_accs = checkpoint['train_accs']
# #     model.valid_losses = checkpoint['valid_losses']
# #     model.valid_accs = checkpoint['valid_accs']
        
#     if vocab_path:
#         vocabs = torch.load(vocab_path, lambda storage, loc: storage)
#         return model, checkpoint, all_params, vocabs
    
#     return model, checkpoint, all_params


def load_model(model_path):
    """
    load high-level TCN object
    """
    tcn = torch.load(model_path, map_location=lambda storage, loc: storage)
    return tcn
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    adopted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, model_dir, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_dir = model_dir

    def __call__(self, val_loss, model, model_object=None):

        score = -val_loss
        
        if np.isnan(val_loss):
            print('loss turns to nan, stop')
            self.early_stop = True
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_object=model_object)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_object=model_object)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_object=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        checkpoint = {}
        checkpoint['model'] = model.state_dict()     
        torch.save(checkpoint, '{}/checkpoint.pt'.format(self.model_dir))
        if model_object:
            print("saving checkpoint!")
            model_object.save(self.model_dir)
        self.val_loss_min = val_loss
        

class LossWeighing:
    def __init__(self, wtype='reconstruction', ep_threshold=10):
        try:
            self.weight = float(wtype)
        except:
            self.wtype = wtype
            self.ep_threshold = ep_threshold
       
    
    def __call__(self, ep):
        if hasattr(self, 'weight'):
            return self.weight
        
        elif self.wtype == 'half_linear':
            return min((ep+self.ep_threshold)/2/self.ep_threshold, 1)
        
        elif self.wtype.lower == 'language':
            return 0
        
        else:
            return 1

        

def parse_meta(meta, seq_len, return_sequences=True, min_length=1):
    seq_lengths = list(map(len, meta))
    assert(seq_lengths == seq_len.tolist())
    
    if not return_sequences:
        meta = list(map(lambda x: x[-1:], meta))
    else:
        meta = list(map(lambda x: x[min_length-1:], meta))
    return np.concatenate(list(filter(lambda x: len(x) > 0, meta)), axis=0)

    
    
def parse_pred(pred, seq_len, return_sequences=True, min_length=1):
    """
    given a pred/target tensor with shape b * s * f, and a list of seq_lens,
    extract the non-padded timesteps, end up with a tensor with shape sum(seq_lens) * f
    """
    shape = pred.shape
    if len(shape) == 3: # pred is probability
        b, s, f = shape
    elif len(shape) == 2: # pred is target
        b, s = shape
    
    preds = []
    
    start = min_length -1
    for bi in range(b):
        si = int(seq_len[bi].item())
        if return_sequences:
            preds.append(pred[bi, start:si, :])
        else:
            preds.append(pred[bi, si-1, :])
    return torch.cat(preds).flatten()


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def build_summary(target, pred, percentiles):
    """ cumulative """
    df = []
    target = pd.Series(target == 1)
    pred = pd.Series(pred)
    for thresh, pctl in [(np.percentile(pred, pctl), pctl) for pctl in percentiles]:
        pred_tmp = pred >= thresh
        rep = classification_report(y_true=target, y_pred=pred_tmp, output_dict=True)
        conf = confusion_matrix(y_true=target, y_pred=pred_tmp)
        df.append([pctl, thresh, (1 - rep['True']['precision']) * 100, rep['True']['recall'] * 100,
                  sum(conf[:, 1]), conf[1][1], conf[0][1], conf[1][0]])
    return pd.DataFrame(df, columns=['Percentile', 'Threshold', 'False Positive Rate (%)', 
                                     'Fraud Capture Rate (%)', '#Above Threshold', 
                                     '#Fraudulent Above Threshold', '#Good Above Threshold',
                                     '#Fraudulent Below Threshold'])

# pctls = np.linspace(90, 100, 10, endpoint=False)
# build_summary(target, pred, pctls)



################################################
#             Printing and Logging
################################################


def print_param(title=None, description=None):
    if title:
        print("-" * 30 + " " + str(title) + " " + "-" * 30)
    if description:
        print(str(description))
    
    
def log_status(log_path, status, param=None, init=True, verbose=1):
    """
    @param log_path: path of the log
    @param status: the message to log and/or print
    @param param: dictionary or object that need to be logged along with status
    @param init: whether it is the initial call. 
                   - True: create a new log 
                   - False: append to existing log
    @param verbose: whether to print the status and param when logging. 
                   - 1: print
                   - 0: do not print
    """
    if init:
        output_file = open(log_path, "w")
    else: # updating log
        output_file = open(log_path, "a")
        
    message = str(status)
    if verbose > 0:
        print(message)
    output_file.write(message + "\n")
    if param:
        message = str(param)
        if verbose > 0:
            print(message)
        output_file.write(message + "\n")
    output_file.close()
    

################################################
#                  Timer
################################################

    
import time

class Timer: 
    def __init__(self, unit='M'):
        self.global_start = time.time()
        self.cur_start = time.time()
        self.is_timer_paused = False
    
        assert(unit.lower() in ['h', 'm', 's'])
        self.unit = unit.lower()
        if self.unit == 'h':
            self.denum = 3600
        elif self.unit == 'm':
            self.denum = 60
        elif self.unit == 's':
            self.denum = 1
        
        
    def time(self):
        """
        return time in since the last time calling .time() after removing paused time
        """
        if self.is_timer_paused:
            self.resume()
            self.is_timer_paused = False
        time_past = time.time() - self.cur_start
        self.cur_start = time.time()
        
        return round(time_past/self.denum, 2)
       
    
    def time_since_start(self):
        if self.is_timer_paused:
            self.resume()
            self.is_timer_paused = False
        
        time_past = time.time() - self.global_start
        return round(time_past/self.denum, 2)
    
    
    def pause(self):
        self.is_timer_paused = True
        self.paused_timer = time.time()
        return None
    
    
    def resume(self):
        self.is_timer_paused = False
        time_passed = time.time() - self.paused_timer
        self.global_start += time_passed
        self.cur_start += time_passed
        return None
    
    
    def reset(self):
        self.global_start = time.time()
        self.cur_start = time.time()
    
    
################################################
#        LossGenie: Loss utility fn
################################################

        
import torch.nn as nn 

class LossGenie(object):

    def __init__(self, criterion, **kwargs):
        ''' 
        @param criterion: 'str' indicating the criterion to select from torch.nn.
            e.g. BCEWithLogitsLoss, CrossEntropyLoss
        @param **kwargs: dictionary of key-value pairs that the selected criterion would need
        
        Just like a diaper genie, only for loss. This wrapper object provides cumulative 
        functionalities the loss functions.
        
        criterion is automatically set to training mode when initialized.

        Initialization

        ------------- Example 1: BCEWithLogitsLoss -------------

            kwargs = {'pos_weight': torch.Tensor([10]).to(args.device), 
                      'reduction': 'mean'}
            loss = LossGenie("BCEWithLogitsLoss", **kwargs)
            criterion = nn.BCEWithLogitsLoss(**kwargs)

            loss(output, target) == criterion(output, target)

        ------------- Example 2: CrossEntropyLoss -------------

            kwargs = {}
            loss = LossGenie("CrossEntropyLoss", **kwargs)
            criterion = nn.CrossEntropyLoss(**kwargs)

            loss(output, target) == criterion(output, target)    

        '''
        self.running_loss_train = 0.0
        self.running_loss_test = 0.0
        self.n_batch_train = 0.0
        self.n_batch_test = 0.0
        self.train_losses = []
        self.test_losses = []
        self.criterion = getattr(nn, criterion)(**kwargs)
        self.training = True
        
        
    def __call__(self, out, tgt):
        '''
        Compute loss given out and tgt. 
    
        out and tgt must satisfy required formats for the selected criterion
        
        @param out: tensor
        @param tgt: tensor
        '''
        loss = self.criterion(out, tgt)
        if self.training:
            self.running_loss_train += loss.item()
            self.n_batch_train += 1
        else:     
            self.running_loss_test += loss.item()
            self.n_batch_test += 1
        return loss
    
    
    def get_running_loss(self, reduction='mean'):
        '''
        get the running loss, with reduction by mean or sum
        '''
        if self.training and self.n_batch_train < 1:
            print("Haven't processed any training batch")
            return None
        elif not self.training and self.n_batch_test < 1:
            print("Haven't processed any testing batch")
            return None
        if reduction == 'mean':
            loss = self.running_loss_train / self.n_batch_train if self.training else self.running_loss_test / self.n_batch_test

        elif reduction == 'sum':
            loss = self.training * self.running_loss_train + (1-self.training) * self.running_loss_test
        else:
            raise ValueError("unknown parameter for reduction. (mean or sum)")

        if self.training:
            self.train_losses.append(loss)
        else:
            self.test_losses.append(loss)
        return loss
    
    
    def get_last_valid_loss(self):
        if len(self.test_losses) == 0:
            return None
        return self.test_losses[-1]
    
    
    def reset(self):
        '''
        resetting the running loss for the current task (train, test)
        '''
        if self.training:
            self.running_loss_train = 0.0
            self.n_batch_train = 0.0
        else:
            self.running_loss_test = 0.0
            self.n_batch_test = 0.0
            
    
    def reset_train(self):
        self.running_loss_train = 0.0
        self.n_batch_train = 0.0
        
        
    def reset_test(self):
        self.running_loss_test = 0.0
        self.n_batch_test = 0.0
    
    
    def train(self):
        self.training = True
        
    
    def eval(self):
        self.training = False
        
        
        
###############################################################
#                Inference Formatter
###############################################################
        
        
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, accuracy_score


class InferenceFormatter:
    def __init__(self):
        self.probs = []
        self.metas = []
        self.len = 0
        self.y_prob_type = None
        self.meta_type = None
        
        
    def collect(self, y_prob, meta=None):        
        """
        @param y_prob : 1d array-like, or label indicator array / sparse matrix
                   Predicted probability returned by a classifier.
        
        @param meta : 1d array-like, or label indicator array / sparse matrix
                   meta data.

        """
        if self.y_prob_type is None: 
            self.y_prob_type = self.check_type(y_prob)
        if meta is not None and self.meta_type is None:
            self.meta_type = self.check_type(meta)
            
        y_prob = self.cast(y_prob, self.y_prob_type)
        self.probs.extend(y_prob)
        
        if meta is not None:
            self.metas.extend(np.concatenate(meta).tolist())
        
        self.len += len(y_prob)
        
        
    def get_inference(self, meta_columns=None, return_sequences=True):
        probs = np.array(self.probs)
        if meta_columns:
            idx = np.array(self.metas)            
            results = pd.DataFrame(np.concatenate([idx.reshape(-1, len(meta_columns)), probs.reshape(-1,1)], axis=1))
            results.columns = meta_columns + ['tcn_pred']
        else:
            results = probs
        return results
    
        
    def reset(self):
        self.probs = []
        self.metas = []
        self.len = 0
        self.y_prob_type = None
        self.meta_type = None
        
        
    def check_type(self, obj):
        """
        check whether the object type is list, numpy, or torch
        """
        type_ = str(type(obj))
        for t in ["torch", "numpy", "list"]:
            if t in type_: return t
        raise TypeError("Unknown type, only takes in torch, numpy, or list")

    
    def cast(self, obj, obj_type):
        """
        cast obj from self.type to list
        
        self.type in [list, numpy, torch]
        """
        if obj_type == "list":
            return obj
        elif obj_type == "numpy":
            return obj.tolist()
        elif obj_type == "torch":
            return obj.cpu().detach().tolist()
        else:
            raise ValueError("unknown type, only casts numpy and torch to list")
        
    

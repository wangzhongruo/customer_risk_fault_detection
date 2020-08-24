import re
import dill
import numpy as np
import torch
import copy
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class myDataset(Dataset):
    
    def __init__(self, data_object, is_train, inference, train_dataset=None, load_vocabs=None):
        """
        self.data: list of arrays of features
        
        if train_dataset exist, load vocab from train_dataset
        """
        self.is_train = is_train
        self.inference = inference
        
        if is_train or inference:
            # data is a list of arrays
            self.data = data_object['train']

        else:
            # data is a list of arrays
            self.data = data_object['test']

        self.features_num = data_object['features_num']
        self.features_cat = data_object['features_cat']
        self.features_meta = data_object['features_meta']        
        self.features = data_object['features'] # all features
        self.col2i = copy.deepcopy(data_object['features_idx'])
        self.source_features = self.features_num + self.features_cat  # features model will use (May or may not include target?)        
        self.len = len(self.data)
        

        if load_vocabs is not None:
            print("loading vocabs....")
            self.vocab_lens = load_vocabs['vocab_lens']
            self.w2i_dicts = load_vocabs['w2i_dicts']
            self.i2w_dicts = load_vocabs['i2w_dicts']
        elif is_train:
            if inference:
                print("making inference with randomly generated embeddings. Please provide vocab for consistent results.")
            self.build_vocab()
        else:
            self.load_vocab(train_dataset)
            
        self.encode_source()
        
        self.source = self.select_cols(self.data, self.source_features, self.col2i) # train 
        self.meta = self.select_cols(self.data, self.features_meta, self.col2i)

        if not inference:
            self.features_target = data_object['features_target']
            self.target = self.select_cols(self.data, self.features_target, self.col2i) # list of targets
    
        
    def build_vocab(self):
        self.vocab_lens = {}
        self.w2i_dicts = {}
        self.i2w_dicts = {}
        for f in self.features_cat:
            col = self.col2i[f]
            vocab_iter = (map(lambda d_array: d_array[:, col], self.data))
            vocab = set()
            for v in vocab_iter:
                vocab = vocab.union(v)
            is_None = None in vocab
                                        
            vocab = list(vocab)
            vocab = sorted(list(filter(lambda b:b is not None, vocab))) # sort the vocab list
            
            if is_None:
                vocab = np.append(vocab, 'None')
            
            self.vocab_lens[f] = len(vocab) + 3
            self.w2i_dicts[f] = dict(zip(vocab, range(3, len(vocab)+3)))
            self.w2i_dicts[f]['<pad>'] = 0
            self.w2i_dicts[f]['<unk>'] = 1
            self.w2i_dicts[f]['<eos>'] = 2

            self.i2w_dicts[f] = dict(zip(range(3, len(vocab)+3), vocab))
            self.i2w_dicts[f][0] = '<pad>'
            self.i2w_dicts[f][1] = '<unk>'
            self.i2w_dicts[f][2] = '<eos>'
            
            print('feature ({}) vocabulary size: {} '.format(f, self.vocab_lens[f]))

            
    def load_vocab(self, train_dataset):
        self.vocab_lens = train_dataset.vocab_lens
        self.w2i_dicts = train_dataset.w2i_dicts
        self.i2w_dicts = train_dataset.i2w_dicts
    
    
    def encode_source(self):
        """
        index values in self.source: map to ints
        """
        
        n_unk = 0
        
        def w2i(feature_w2i, token, vocab, n_unk):
            if token not in vocab:
                n_unk += 1
            return feature_w2i[token] if token in vocab else feature_w2i['<unk>']
        
        for col in self.features_cat:
            col_idx = self.col2i[col]
            vocab = self.w2i_dicts[col].keys()
            mapper = np.vectorize(lambda x: w2i(self.w2i_dicts[col], x, vocab, n_unk))
            
            for datum in self.data:
#                 import pdb; pdb.set_trace()
                datum[:, col_idx] = mapper(datum[:, col_idx])
        if not self.is_train:
            print('num unknown tokens in test set: ',n_unk)
            
    
    def select_cols(self, data, features, col2i):
        cols = [col2i[f] for f in features]
        result = list(map(lambda d_array: d_array[:, cols], data))
        return result
        
        
    def __getitem__(self, index):
        if self.inference: # when encoding 
            datum = (self.source[index], self.meta[index])
        else: # when training and validating
            datum = (self.source[index], self.target[index], self.meta[index])
        return datum
    
    
    def __len__(self):
        return self.len
        
        
def collate_fn_pad(data, pad_after=True):
    """
    Creates mini-batch tensors from 
    * if unsupervised: list of item - src
    * if supervised: the list of tuples - tuple(src, tgt).
    
    We should build a custom collate_fn to pad variable length sequences and 
    then sorted by length
    
    Args:
        data: list of array (a data point)
        
    Returns:
        packed padded seq
    """        

    # revert seq_tensor for target_tensor
    source, target, meta = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor(list(map(len, source)))
#     print("-"*80)
#     print("target: ")
#     print(target)
#     print("seq_lengths: ")
#     print(seq_lengths)
                        
    # pad source seq in the front and target in the end, since we reconstruct in the reverse order
    src_seq, _ = pad(source, seq_lengths, pad_after=pad_after)
    tgt_seq, _ = pad(target, seq_lengths, pad_after=pad_after)
    
    return src_seq, tgt_seq, seq_lengths, meta 
    

def collate_fn_pad_inference(data, pad_after=True):
    """
    Creates mini-batch tensors from 
    * if unsupervised: list of item - src
    * if supervised: the list of tuples - tuple(src, tgt).
    
    We should build a custom collate_fn to pad variable length sequences and 
    then sorted by length
    
    Args:
        data: list of array (src, meta)
        
    Returns:
        packed padded seq
    """        

    # revert seq_tensor for target_tensor
    source, meta = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor(list(map(len, source)))
    
    # pad source seq in the front and target in the end, since we reconstruct in the reverse order
    src_seq, _ = pad(source, seq_lengths, pad_after=pad_after)

    return src_seq, seq_lengths, meta
    

def pad(source, seq_lengths, pad_after=True): # batch_size x sequence_length_max x feature_dim 
    max_seq_len = seq_lengths.max()
    seq_tensor = Variable(torch.zeros((len(source), max_seq_len, len(source[0][0])))).long()
    # pad input tensor
    for idx, seq in enumerate(source):
        seq_len = seq_lengths[idx]
        if pad_after:
            seq_tensor[idx, :seq_len] = LongTensor(np.asarray(seq).astype(int))
        else: 
            # pad before
            seq_tensor[idx, max_seq_len-seq_len:] = LongTensor(np.asarray(seq).astype(int))
            
    # sort by seq_length
#     seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#     seq_tensor = seq_tensor[perm_idx]
    return seq_tensor, seq_lengths
    

def get_loader(data_object, batch_size=2, pad_after=True, vocabs=None, inference=False):
    train_dataset = myDataset(data_object, is_train=not inference, load_vocabs=vocabs, inference=inference)   
    
    collate_wrapper = lambda d: collate_fn_pad(d, pad_after=pad_after) if not inference else collate_fn_pad_inference(d, pad_after=pad_after)
    
    print("shuffling dataset: ", not inference)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=(not inference),
                              collate_fn=collate_wrapper)
    
    if 'test' in data_object:
        test_dataset = myDataset(data_object, is_train=False, train_dataset=train_dataset, inference=inference)
        test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_wrapper)

        return train_loader, test_loader    
    return train_loader

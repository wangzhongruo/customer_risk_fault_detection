import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, accuracy_score, average_precision_score


class Metrics:
    """
    General Metrics class for model training with pytorch
    
    Example: ROC_AUC
    
    ======================================================================
    (1) define class
    
    class ROC_AUC(Metrics):
        def __init__(self):
            super(ROC_AUC, self).__init__()

        def __call__(self):
            return roc_auc_score(self.targets, self.preds)
            
    ======================================================================
    (2) ROC_AUC.collect() truth and predicted values
    
    pred = [random.random() for i in range(10)]
    target = [random.randint(0,1) for i in range(10)]

    auc.collect(y_true=target,y_pred=pred)
    

    ======================================================================
    (3) ROC_AUC(): call the object to obtain current value
    value = auc()

    ======================================================================
    (4) ROC_AUC.clear(): to clear the stored predictions
    """
    def __init__(self):
        self.targets = []
        self.probs = []
        self.len = 0
        self.y_true_type = None
        self.y_prob_type = None
        
        
    def collect(self, y_prob, y_true=None):
        """
        @param y_true : 1d array-like, or label indicator array / sparse matrix
                   Ground truth (correct) target values.

        @param y_prob : 1d array-like, or label indicator array / sparse matrix
                   Predicted probability returned by a classifier.
        
        """
        if self.y_true_type is None: 
            self.y_true_type = self.check_type(y_true)
            self.y_prob_type = self.check_type(y_prob)
            
        y_true = self.cast(y_true, self.y_true_type)
        y_prob = self.cast(y_prob, self.y_prob_type)
        self.targets.extend(y_true)
        self.probs.extend(y_prob)
        self.len += len(y_true)
        
        
    def reset(self):
        """ clear stored values """
        self.targets = []
        self.probs = []
        self.len = 0
        
        
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
        
        
    def __call__(self):
        raise NotImplementedError("Need to implement for each individual metric")
        
        

class BinaryClfMetrics(Metrics):
    def __init__(self):
        super(BinaryClfMetrics, self).__init__()
        
        
        
    def build_summary(self, target, pred, percentiles):
        """ 
        Summary DataFrame of predictions
        """
        df = []
        target = pd.Series(np.array(target) == 1)
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
    
    
    def sigmoid(self, x):
        sigm = 1. / (1. + np.exp(-np.array(x)))
        return sigm
    
    
    def roc_auc(self):
        """
        return the roc auc score
        """
        if not hasattr(self, 'roc_auc_list'):
            self.roc_auc_list = []
        auc = roc_auc_score(self.targets, self.probs)
        self.roc_auc_list.append(auc)
        return auc
    
    
    def average_precision(self):
        """
        return the average_precision score
        """
        if not hasattr(self, 'average_precision_list'):
            self.average_precision_list = []
        ap = average_precision_score(self.targets, self.probs)
        self.average_precision_list.append(ap)
        return ap
    
    
    def top_k_percentile(self, k=10, step=1, decimal=2):
        """
        @param k: top k percentile to look at
        @param step: step size. i.e. 
        
        e.g. 
            k = 2, step = 0.5 ->
            
            percentile
                98
                98.5
                99
                99.5 
        """
        assert(0 < k <= 100)
        topkpctls = np.linspace(100-k, 100, int(k/step), endpoint=False)
        return self.build_summary(self.targets, self.probs, topkpctls).round(decimal)
    
    
    def top_10_percentile(self):
        top10pctls = np.linspace(90, 100, 10, endpoint=False)
        return self.build_summary(self.targets, self.probs, top10pctls).round(2)
 

    def top_20_percentile(self):
        top20pctls = np.linspace(80, 100, 20, endpoint=False)
        return self.build_summary(self.targets, self.probs, top20pctls).round(2)
    
    
    def accuracy(self):
        """
        return accuracy score
        """
        if not hasattr(self, 'accuracy_list'):
            self.accuracy_list = []
        preds = np.array(self.probs) >= 0.5
        acc = accuracy_score(self.targets, preds)
        self.accuracy_list.append(acc)
        return acc
    
    
    def get_metrics(self):
        """
        Get all previously computed metrics!
        """
        result = {}
        for k, v in vars(self).items():
            if 'list' in k:
                result[k] = v
        return result
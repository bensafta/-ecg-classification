# src/utils/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    
    try:
        auc = roc_auc_score(np.eye(len(np.unique(y_true)))[y_true], 
                            np.eye(len(np.unique(y_pred)))[y_pred], 
                            multi_class='ovr')
    except:
        auc = 0.0
        
    accuracy = np.mean(y_true == y_pred)
    return accuracy, sensitivity, specificity, auc

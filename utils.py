import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *

def plot_roc_curve(fpr, tpr, auc_score, plotter, save_path):
    figure = plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plotter.figure_plot('roc_curve', figure, 0)
    plt.close(figure)

def make_model_name_dir(default_path):
    model_name_list = ['STL', 'FSMTL', 'SPMTL', 'ADMTL', 'ADMTL3']
    for model_name in model_name_list:
        if not os.path.exists(os.path.join(default_path, model_name)):
            os.mkdir(os.path.join(default_path, model_name))
            

def get_auc_score(pred, true):
    fpr, tpr, thresholds = roc_curve(true, pred)
    auc_score = auc(fpr, tpr)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return auc_score, best_threshold, [fpr,tpr,thresholds]

def get_precision_recall_ap(pred, true):
    precision, recall, thresholds = precision_recall_curve(true, pred)
    ap = average_precision_score(true, pred)
    return precision, recall, ap
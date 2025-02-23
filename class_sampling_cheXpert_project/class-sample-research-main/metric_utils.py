import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import metrics
import numpy as np

import torch
import torch.nn as nn

import inference
    
def triplet_loss(test_loader, network): 
    
    test_losses = inference.run_inference_triplet_loss(test_loader, network)
    print(f'\nTest set: Avg. loss: {test_losses[-1]}')  

    return test_losses 
    
    
def auc_sigmoid(test_loader, network, embeddings=False):
    
    test_losses, y_preds, y_true = inference.run_inference_sigmoid(test_loader, network, embeddings=embeddings)
        
    network_auc = auc(y_preds, y_true)
    
    print(f'\nTest set: Avg. loss: {test_losses[-1]}, AUC: {network_auc}\n')   
    
    return test_losses, network_auc


def auc_softmax(test_loader, network, embeddings=False):
    
    test_losses, y_preds, y_true = inference.run_inference_softmax(test_loader, network, embeddings=embeddings)
    
    network_auc = multiclass_auc(y_preds, y_true)
    
    
    print(f'\nTest set: Avg. loss: {test_losses[-1]}, AUC: {network_auc}\n') 
    
    return test_losses, network_auc


def accuracy_sigmoid(test_loader, network): 
    test_losses, y_preds, y_true = inference.run_inference_sigmoid(test_loader, network)
    
    acc = accuracy(y_preds, y_true)
    
    print(f'\nTest set: Avg. loss: {test_losses[-1]}, Accuracy: {acc}%\n')

    return test_losses, acc


def accuracy_softmax(test_loader, network):
    test_losses, y_preds, y_true = inference.run_inference_softmax(test_loader, network)
    acc = accuracy(y_preds, y_true)
    
    print(f'\nTest set: Avg. loss: {test_losses[-1]}, Accuracy: {acc}%\n')
    
    return test_losses, acc


def accuracy(y_pred, y_true):
    return torch.sum(torch.stack(y_pred).squeeze()==torch.stack(y_true))/len(y_pred)

def auc(y_pred, y_true):
    return metrics.roc_auc_score(y_true, y_pred)

def multiclass_auc(y_pred, y_true): 
    return metrics.roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr') 


def get_confidence_interval(y_pred, y_true, verbose=True, metric_fn=accuracy):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        
        score = metric_fn(y_pred[indices], y_true[indices])
        bootstrapped_scores.append(score)


    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    metric = metric_fn(y_pred, y_true) 
    

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    if verbose:
        print("Accuracy: {} [{:0.3f} - {:0.3}]".format(
          accuracy, confidence_lower, confidence_upper))
    return confidence_lower, confidence_upper, metric
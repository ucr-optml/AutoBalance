
from collections import deque

import numpy as np

import torch

# def topk_errors(preds, labels, ks):
#     """Computes the top-k error for each k."""
#     err_str = "Batch dim of predictions and labels must match"
#     assert preds.size(0) == labels.size(0), err_str
#     # Find the top max_k predictions for each sample
#     _top_max_k_vals, top_max_k_inds = torch.topk(
#         preds, max(ks), dim=1, largest=True, sorted=True
#     )
#     # (batch_size, max_k) -> (max_k, batch_size)
#     top_max_k_inds = top_max_k_inds.t()
#     # (batch_size, ) -> (max_k, batch_size)
#     rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
#     # (i, j) = 1 if top i-th prediction for the j-th sample is correct
#     top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
#     # Compute the number of topk correct predictions for each k
#     topks_correct = [top_max_k_correct[:k, :].view(-1).float().sum() for k in ks]
#     return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]

def topk_corrects(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].view(-1).float().sum() for k in ks]
    return topks_correct

def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params 
    print("-" * 50)
    print("Total number of parameters: {:.2e}".format(total_num_params))

def smooth(scalars, weight=0.8):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
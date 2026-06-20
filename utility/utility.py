import os
import numpy as np
from sklearn.metrics import roc_curve
import torch
from torch.utils.data import Dataset

class MultiDatasetLoader(Dataset):
    def __init__(self, file_list_with_paths):
        self.file_list = file_list_with_paths

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        feature, label = torch.load(self.file_list[idx], weights_only=True)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return feature, label
    
def get_pt_files(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pt')]

def collect_labels(dataloader):
    all_labels = []
    sample_feature = None
    
    for features, labels in dataloader:
        all_labels.append(labels.clone()) # Keep the tiny label tensors
        if sample_feature is None:
            sample_feature = features[0].clone() # save one example to inspect shape
            
    all_labels = torch.cat(all_labels, dim=0)
    return sample_feature, all_labels

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    if len(thresholds) > 0:
        thresholds[0] = max(scores)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))

    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return eer * 100, thresholds[eer_idx]

def calculate_gating_balance_loss(gating_weights):
    """
    gating_weights: Tensor of shape [batch_size, num_experts]
    Returns a scalar loss penalizing unequal expert utilization.
    """
    # Calculate the mean allocation for each expert across this entire batch
    mean_weights = torch.mean(gating_weights, dim=0) # Shape: [num_experts]
    
    # Calculate the variance and mean squared to find the coefficient of variation
    variance = torch.var(mean_weights, unbiased=False)
    mean_squared = torch.mean(mean_weights) ** 2
    
    # Return the balance penalty (lower is more evenly distributed)
    return variance / (mean_squared + 1e-10)
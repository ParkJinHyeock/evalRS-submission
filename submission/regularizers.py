import torch
import numpy as np
# ref: https://arxiv.org/pdf/1705.08804.pdf

def U_abs(group, loss):
    ids=torch.unique(group) #array of unique ids
    group_mean=[torch.mean(loss[group==i]) for i in ids]
    total_mean=torch.mean(loss)
    group_diff = torch.stack([abs(i-total_mean) for i in group_mean])
    return torch.mean(torch.sqrt(torch.pow(group_diff, 2)))
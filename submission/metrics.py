import torch
import torch.nn as nn
import numpy as np


def Hit(gt_items, pred_items):
    pred_items = pred_items.cpu().numpy()
    gt_items = gt_items.cpu().numpy()
    count = 0
    for i, gt_item in enumerate(gt_items):
        if gt_item in pred_items[i,:]:
            count += 1
    return count/len(gt_items)


def ndcg(gt_items, pred_items):
    pred_items = pred_items.cpu().numpy()
    gt_items = gt_items.cpu().numpy()
    ndcg = 0
    for i, gt_item in enumerate(gt_items):
        if gt_item in pred_items[i,:]:
            index = np.where(pred_items[i,:] == gt_item)[0][0]
            ndcg += np.reciprocal(np.log2(index+2))

    return ndcg/len(gt_items)

def mrr(gt_items, pred_items):
    pred_items = pred_items.cpu().numpy()
    gt_items = gt_items.cpu().numpy()
    mrr = 0
    for i, gt_item in enumerate(gt_items):
        if gt_item in pred_items[i,:]:
            index = np.where(pred_items[i,:] == gt_item)[0][0]
            mrr += np.reciprocal(index+1)
    return mrr/len(gt_items)


if __name__ == '__main__':
    topk = 100 
    user_num = 1024
    gt_item = torch.ones([user_num])
    pred_item = torch.ones([user_num, topk])
    print(Hit(gt_item, pred_item))
    print(ndcg(gt_item, pred_item))
    print(mrr(gt_item, pred_item))
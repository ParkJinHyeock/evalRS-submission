import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from submission.loader import BPR_Dataset
from submission.model import *
from submission.metrics import *
from submission.regularizers import *

from time import time
from tqdm import tqdm
import os
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.autograd import Variable

def train(train_dataset, test_dataset, model_type, num_user, num_item, **kwargs):
    args = kwargs['args']
    user_group_dict, item_group_dict = kwargs['user_group_dict'], kwargs['item_group_dict']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ensemble = kwargs['use_ensemble']

    if model_type == 'BPR':
        if use_ensemble:
            model = BPR(num_user, num_item, 64).to(device)    
        else:
            model = BPR(num_user, num_item,args.dim).to(device)    

    if model_type == 'VAE':
        if use_ensemble:
            model = VAE(num_user, num_item, args.dim, not(args.AE_user)).to(device)
        else:
            model = VAE(num_user, num_item, args.dim, args.AE_user).to(device)
    if use_ensemble and (model_type == 'BPR'):
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if model_type == 'BPR':
        if use_ensemble:
            epochs = 10
        else:
            epochs = args.epoch
        for epoch in range(epochs):
            positive_list = []
            model.train()
            with tqdm(train_dataset, unit='batch') as tepoch:
                for positive, negative_users, user_track_count in tepoch:
                    tepoch.set_description(f"Train Epoch {epoch}")
                    negative_items = torch.randperm(num_item, device=device)[:positive.shape[0]]
                    negative = torch.stack([negative_users.to(device), negative_items], dim=1)
                    optimizer.zero_grad()
                    loss = model(positive.to(device), negative.to(device), user_track_count.to(device))
                    loss_masked = loss * torch.squeeze(user_track_count.cuda())
                    loss_sum = -torch.sum(torch.log(loss_masked).to(device)) 
                    loss_sum.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.mean().item())
                    positive_list.append(positive)
                positive_list = torch.cat(positive_list, dim=0)
            model.eval()

            with tqdm(test_dataset, unit='batch') as tepoch:
                total_len = 0
                for positive, negative_users, user_track_count in tepoch:      
                    tepoch.set_description(f"Test Epoch {epoch}")
                    total_len += len(positive)
                    negative_items = torch.randperm(num_item, device=device)[:positive.shape[0]]
                    negative = torch.stack([negative_users.to(device), negative_items], dim=1)
                    loss = model(positive.to(device), negative.to(device), user_track_count.to(device)) 
                    tepoch.set_postfix(loss=loss.mean().item())       

    if model_type == 'VAE':
        for epoch in range(args.epoch):
            model.train()
            with tqdm(train_dataset, unit='batch') as tepoch:
                for id, item in tepoch:
                    if args.AE_user:
                        group_ax0 = id.clone().detach().apply_(user_group_dict.get)
                    else:
                        group_ax0 = id.clone().detach().apply_(item_group_dict.get)
                    item = item.to(device)
                    tepoch.set_description(f"Train Epoch {epoch}")
                    optimizer.zero_grad()
                    logits, mu, logvar, _ = model(item)
                    BCE, KLD, log_softmax_var = model.loss_function(logits, item, mu, logvar)
                    loss = BCE + args.beta*KLD
                    # fairness regularizer
                    reg = U_abs(group_ax0, torch.sum(log_softmax_var * item, dim=1))
                    if reg:
                        total_loss = loss + args.gamma*reg
                    else:
                        total_loss = loss
                    total_loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

                model.eval()
                with tqdm(test_dataset, unit='batch') as tepoch:
                    total_len = 0
                    for id, item in tepoch:
                        item = item.to(device)
                        tepoch.set_description(f"Test Epoch {epoch}")
                        recon, mu, logvar, z = model(item)
                        BCE, KLD, log_softmax_var = model.loss_function(recon, item, mu, logvar)
                        loss = BCE + args.beta*KLD
                        tepoch.set_postfix(loss=loss.item())
                        tepoch.set_postfix(loss=loss.mean().item())       
    return model
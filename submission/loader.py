import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import sys
sys.path.append('../')
from submission.utils import *


class BPR_Dataset(Dataset):
    def __init__(self, train_df, lambda_pop):
        
        # Making dictionary
        self.train_df = train_df
        self.user_dict_reverse = dict(enumerate(train_df['user_id'].unique()))#.to_numpy()))
        self.item_dict_reverse = dict(enumerate(train_df['track_id'].unique()))#.to_numpy()))
      
        self.user_dict = {v: k for k, v in self.user_dict_reverse.items()}
        self.item_dict = {v: k for k, v in self.item_dict_reverse.items()}

        # train_df = train_df[train_df['user_track_count'] > 1]
        self.user_interactions = train_df[['user_id']]#.to_numpy()
        self.item_interactions = train_df[['track_id']]#.to_numpy()

        self.user_interactions = torch.from_numpy(np.squeeze(np.vectorize(self.user_dict.get)(self.user_interactions)))
        self.item_interactions = torch.from_numpy(np.squeeze(np.vectorize(self.item_dict.get)(self.item_interactions)))

        interactions = np.stack([self.user_interactions, self.item_interactions], axis=1)
        self.interactions = torch.from_numpy(interactions)
        self.track_count = torch.from_numpy(train_df[['user_track_count']].to_numpy())
        self.track_count = (self.track_count == 1).to(torch.float64) * lambda_pop + ((self.track_count != 1).to(torch.float64))         
        # create user2group dict
        self.user_group_dict = get_group(self.train_df, 'user_id', self.user_dict)
        # create item2group dict
        self.item_group_dict = get_group(self.train_df, 'track_id', self.item_dict)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        """        
        return
        positive_sample: [user i, item j]
                         The first element contains the user index and the second column contains that of the item.
                         It implies that a user i(first element) has an interaction with an item j(second element).
        negative_users:  [user i]
                         The element is equal to the first element of the positive sample to keep the user information when sampling a negative sample.
        """
        from time import time
        positive_sample = self.interactions[index]
        negative_users = self.interactions[index][0]
        user_track_count = self.track_count[index]

        return positive_sample, negative_users, user_track_count

class Autorec_Dataset(Dataset):
    def __init__(self, train_df, use_user):
        self.train_df = train_df
        self.use_user = use_user
        self.user_dict_reverse = dict(enumerate(train_df['user_id'].unique()))
        self.item_dict_reverse = dict(enumerate(train_df['track_id'].unique()))       
        self.artist_dict_reverse = dict(enumerate(train_df['artist_id'].unique()))        

        self.user_dict = {v: k for k, v in self.user_dict_reverse.items()}
        self.item_dict = {v: k for k, v in self.item_dict_reverse.items()}
        self.artist_dict = {v: k for k, v in self.artist_dict_reverse.items()}

        # create user2group dict
        self.user_group_dict = get_group(self.train_df, 'user_id', self.user_dict)
        # create item2group dict
        self.item_group_dict = get_group(self.train_df, 'track_id', self.item_dict)
        # create album2group dict
        self.artist_group_dict = get_group(self.train_df, 'artist_id', self.artist_dict)

        self.train_df['user_id'] = self.train_df['user_id'].map(self.user_dict)
        self.train_df['track_id'] = self.train_df['track_id'].map(self.item_dict)   
        self.train_df['artist_id'] = self.train_df['artist_id'].map(self.artist_dict)   
        self.train_df['artist_group'] = self.train_df['artist_id'].map(self.artist_group_dict)
        
        self.item_artist_dict = self.train_df.set_index('track_id').to_dict()['artist_group']
        self.train_df['user_track_count'] = (self.train_df['user_track_count'] > 0) * 1
        temp = self.train_df[['user_id', 'track_id', 'user_track_count']].to_numpy().astype('float32')
        csr = csr_matrix((temp[:,2],(temp[:,1], temp[:,0])), shape=(len(self.item_dict), len(self.user_dict)))
        self.matrix = torch.from_numpy(csr.toarray()).to(torch.float32)
        if self.use_user:
            self.matrix = self.matrix.T #[UxI]

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        return index, self.matrix[index]

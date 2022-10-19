from concurrent.futures import process
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from reclist.abstractions import RecModel
from submission.loader import *
from submission.model import *
from submission.trainer import *
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from reclist.utils.train_w2v import train_embeddings
import gensim
from multiprocessing import Pool
from functools import partial
from torch.utils.data import Subset


class MyModel(RecModel):
    def __init__(self, users:pd.DataFrame, items: pd.DataFrame, top_k: int=100, **kwargs):
        super(MyModel, self).__init__()
        self.args = kwargs['args']
        self.model_type = kwargs['args'].model_type
        self.top_k = top_k
        print("Received additional arguments: {}".format(kwargs))
        return
    
    def train(self, train_df: pd.DataFrame):
        import copy
        train_df = train_df.sort_values(by=['user_id','timestamp'])
        #train_df_2 = train_df.copy()
        if (self.model_type == 'BPR'):
            dataset = BPR_Dataset(train_df, self.args.lambda_pop)
        elif (self.model_type == 'AE') or (self.model_type == 'VAE'):
            dataset = Autorec_Dataset(train_df, use_user=self.args.AE_user)

        num_user = len(dataset.user_dict)
        num_item = len(dataset.item_dict)
        train_len = int(0.98*len(dataset))
        valid_len = len(dataset) - train_len

        if self.args.use_group:
            group_dict = dataset.item_artist_dict
            self.zero_group = [key for key in group_dict if group_dict[key] == 0]
            self.one_group = [key for key in group_dict if group_dict[key] == 1]
            self.two_group = [key for key in group_dict if group_dict[key] == 2]
            if not(self.args.AE_user):
                self.three_group = [key for key in group_dict if group_dict[key] == 3]
            self.temp_group = [key for key in dataset.item_group_dict if dataset.item_group_dict[key] == 0]

        train_data, valid_data = torch.utils.data.random_split(dataset, [train_len, valid_len])

        if self.model_type == 'VAE':
            train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
            valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=True, pin_memory=True)
            if self.args.use_group:
                train_loader_0 = DataLoader(Subset(dataset, self.zero_group), batch_size=32, shuffle=True)
                train_loader_1 = DataLoader(Subset(dataset, self.one_group), batch_size=32, shuffle=True)
                train_loader_2 = DataLoader(Subset(dataset, self.two_group), batch_size=32, shuffle=True)
                train_loader_3 = DataLoader(Subset(dataset, self.three_group), batch_size=32, shuffle=True)
                train_loader_4 = DataLoader(Subset(dataset, self.temp_group), batch_size=32, shuffle=True)
                if self.args.use_ensemble:
                    dataset_2 = BPR_Dataset(train_df, self.args.lambda_pop)
                    train_len = int(0.98*len(dataset_2))
                    valid_len = len(dataset_2) - train_len
                    train_data_2, valid_data_2 = torch.utils.data.random_split(dataset_2, [train_len, valid_len])
                    train_dataloader_2 = DataLoader(train_data_2, batch_size=8192, shuffle=True, num_workers=16)
                    valid_dataloader_2 = DataLoader(valid_data_2, batch_size=8192, shuffle=True, num_workers=16)
                    self.dataset_2 = dataset_2

        else:
            train_dataloader = DataLoader(train_data, batch_size=8192, shuffle=True, num_workers=16)
            valid_dataloader = DataLoader(valid_data, batch_size=8192, shuffle=True, num_workers=16)

        self.dataset = dataset

        if not(self.args.use_group):
            self.my_model = train(train_dataloader, valid_dataloader, self.model_type, num_user, num_item, args=self.args,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict, use_ensemble=False)
        
        else:
            self.args_0 = copy.deepcopy(self.args)
            self.args_0.dim = 17
            self.args_0.epoch = 4

            self.args_1 = copy.deepcopy(self.args)
            self.args_1.dim = 17
            self.args_1.epoch = 2

            self.args_2 = copy.deepcopy(self.args)
            self.args_2.dim = 17
            self.args_2.epoch = 2

            self.args_3 = copy.deepcopy(self.args)
            self.args_3.dim = 17
            self.args_3.epoch = 2

            # config for least popular group
            self.args_4 = copy.deepcopy(self.args)
            self.args_4.dim = 15
            self.args_4.epoch = 2


            # each number denotes artist group, and 4 means least popular track group
            self.my_model_0 = train(train_loader_0, valid_dataloader, self.model_type, num_user, num_item, args=self.args_0,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                use_ensemble=False)

            self.my_model_1 = train(train_loader_1, valid_dataloader, self.model_type, num_user, num_item, args=self.args_1,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                use_ensemble=False)                              

            self.my_model_2 = train(train_loader_2, valid_dataloader, self.model_type, num_user, num_item, args=self.args_2,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                use_ensemble=False)

            self.my_model_3 = train(train_loader_3, valid_dataloader, self.model_type, num_user, num_item, args=self.args_3,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                use_ensemble=False)

            self.my_model_4 = train(train_loader_4, valid_dataloader, self.model_type, num_user, num_item, args=self.args_4,
                                user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                use_ensemble=False)

            if self.args.use_ensemble:
                self.my_model = train(train_dataloader_2, valid_dataloader_2, 'BPR', num_user, num_item, args=self.args,
                                    user_group_dict=dataset.user_group_dict, item_group_dict=dataset.item_group_dict,
                                    use_ensemble=False)

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """
        
        This function takes as input all the users that we want to predict the top-k items for, and 
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation 
        would allow for batch predictions of all the target data points.
        
        """
        
        k = self.top_k
        num_users = len(user_ids)
        print(num_users)
        import numpy as np
        user_ids_tensor = torch.from_numpy(np.squeeze(np.vectorize(self.dataset.user_dict.get)(user_ids.to_numpy())))
        from time import time
        start = time()

        if self.args.model_type == 'BPR':
            recommended = self.my_model.predict(user_ids_tensor.cuda(), self.dataset.interactions, 
                                                item_group_dict=self.dataset.item_group_dict, top_k=k)

        else:
            if not(self.args.use_group):
                recommended  = self.my_model.predict(user_ids_tensor.cuda(), self.dataset.matrix, top_k=k)

            else:
                recommended_0 = self.my_model_0.predict(user_ids_tensor.cuda(), self.dataset.matrix[torch.tensor(self.zero_group)], top_k=39)
                recommended_1  = self.my_model_1.predict(user_ids_tensor.cuda(), self.dataset.matrix[torch.tensor(self.one_group)], top_k=20) # 30
                recommended_2  = self.my_model_2.predict(user_ids_tensor.cuda(), self.dataset.matrix[torch.tensor(self.two_group)], top_k=20) # 19
                recommended_3  = self.my_model_3.predict(user_ids_tensor.cuda(), self.dataset.matrix[torch.tensor(self.three_group)], top_k=20) # 20
                recommended_4  = self.my_model_4.predict(user_ids_tensor.cuda(), self.dataset.matrix[torch.tensor(self.temp_group)], top_k=1)

                if self.args.use_ensemble:
                    recommended_BPR  = self.my_model.predict(user_ids_tensor.cuda(), self.dataset_2.interactions, 
                                                        item_group_dict=self.dataset_2.item_group_dict, top_k=k)

                recommended_0 = np.asarray(self.zero_group)[recommended_0]
                recommended_1 = np.asarray(self.one_group)[recommended_1]
                recommended_2 = np.asarray(self.two_group)[recommended_2]
                recommended_3 = np.asarray(self.three_group)[recommended_3]
                recommended_4 = np.asarray(self.temp_group)[recommended_4]

                # Reordering recommendation
                if self.args.use_ensemble:
                    recommended = torch.from_numpy(np.concatenate([recommended_BPR[:,:1], recommended_2[:,:5], recommended_1[:,:5], recommended_3[:,:5], recommended_0[:,:5],
                                                                recommended_2[:,5:], recommended_1[:,5:], recommended_3[:,5:], recommended_0[:,5:-1],
                                                                recommended_4], axis=1))
                else:
                    recommended = torch.from_numpy(np.concatenate([recommended_2[:,:5], recommended_1[:,:5], recommended_3[:,:5], recommended_0[:,:5],
                                                                recommended_2[:,5:], recommended_1[:,5:], recommended_3[:,5:], recommended_0[:,5:],
                                                                recommended_4], axis=1))                

        pred = recommended.cpu().numpy()
        pred = np.vectorize(self.dataset.item_dict_reverse.get)(pred)
        pred = np.concatenate((user_ids[['user_id']].values, pred), axis=1)
        pred = pd.DataFrame(pred, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')
        return pred



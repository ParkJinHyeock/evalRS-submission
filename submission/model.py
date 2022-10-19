import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class BPR(torch.nn.Module):
    def __init__(self, num_user, num_item, embedding_dim=32):
        super(BPR, self).__init__()
        self.user_embedding = nn.Embedding(num_user, embedding_dim, max_norm=True)
        self.item_embedding = nn.Embedding(num_item, embedding_dim, max_norm=True)

    def forward(self, positive, negative, user_track_count):
        positive_user = self.user_embedding(positive[:,0])
        negative_user = self.user_embedding(negative[:,0])

        positive_item = self.item_embedding(positive[:,1])
        negative_item = self.item_embedding(negative[:,1])

        positive_output = torch.sum(positive_user * positive_item, dim=1)
        negative_output = torch.sum(positive_user * negative_item, dim=1)
        subtraction = positive_output - negative_output

        return torch.sigmoid(subtraction)

    def predict(self, user, interactions, item_group_dict, top_k=100):
        # item_list = torch.matmul(self.user_embedding(user), self.item_embedding.weight.T)
        score_list, item_list = [], []
        interactions = interactions.cuda()
        for i in range(user.shape[0]):
            item_embedding_weight = self.item_embedding.weight
            temp = torch.matmul(self.user_embedding(user[i]), item_embedding_weight.T)
            temp[interactions[interactions[:,0] == user[i]][1]] = -10000
            # temp[187855] = -10000
            temp_score, temp_item = torch.topk(temp, top_k)
            score_list.append(temp_score.cpu())
            item_list.append(temp_item.cpu())
            # item_list[i, interactions[interactions[:,0] == user[i]][:,1]] = -10000
        topk_item = torch.stack(item_list)
        return topk_item

class VAE(torch.nn.Module):
    def __init__(self, num_user, num_item, embedding_dim=32, use_user=False, num_classes=4):
        super(VAE, self).__init__()

        # dims should be Three dimension [200, 600, num_item]
        # num_item = num_user
        self.use_user = use_user
        if self.use_user:
            self.num_vector = num_item
        else:
            self.num_vector = num_user
        self.num_item = num_item
        self.num_user = num_user

        temp_q_dims = [self.num_vector , 300, embedding_dim*2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_dims = [embedding_dim, embedding_dim, self.num_vector]
        self.q_dims = [self.num_vector , 300, embedding_dim]

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.drop = nn.Dropout(0.2) # 0.2 
        self.linear = nn.Linear(embedding_dim, num_classes, bias=True)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def predict(self, user, matrix, top_k=100):
        batch_size = 512
        if (matrix.shape[0] % batch_size) != 0:
            remainder = matrix[-(matrix.shape[0] % batch_size):, :]

        temp = torch.empty((matrix.shape[0], matrix.shape[1]))
        with torch.no_grad():
            for i in range(matrix.shape[0] // batch_size):
                temp[batch_size*(i):batch_size*(i+1)] = self.forward(matrix[batch_size*(i):batch_size*(i+1), :].cuda())[0].cpu()
                mask = (matrix[batch_size*(i):batch_size*(i+1)] == 0).cpu()
                temp[batch_size*(i):batch_size*(i+1)] = temp[batch_size*(i):batch_size*(i+1)] * mask 
            if (matrix.shape[0] % batch_size) != 0:
                temp[-(matrix.shape[0] % batch_size):, :] = self.forward(remainder.cuda())[0].cpu()
                mask = (matrix[-(matrix.shape[0] % batch_size):] == 0).cpu()
                temp[-(matrix.shape[0] % batch_size):, :]  = temp[-(matrix.shape[0] % batch_size):, :]  * mask 

        if not(self.use_user):
            temp = temp.T

        temp = temp[user,:]
        _, topk_item = torch.topk(temp, top_k)
        topk_item = topk_item[user,:]
        return topk_item

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def loss_function(self, logits, x, mu, logvar, anneal=1.0):
        # BCE = F.binary_cross_entropy(recon_x, x)
        log_softmax_var = F.log_softmax(logits, 1)
        BCE = - torch.mean(torch.sum(log_softmax_var * x, dim=1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE, KLD, log_softmax_var
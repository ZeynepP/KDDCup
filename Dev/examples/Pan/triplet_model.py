# import sys
# import inspect
import operator

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch import nn

from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.utils import softmax, degree
from torch_geometric.nn import MessagePassing, global_add_pool, GATConv, global_mean_pool, global_max_pool, \
    GlobalAttention, Set2Set
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.nn.pool import TopKPooling, SAGPooling

from torch.utils.data import random_split

from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye

# from collections import OrderedDict

import os
import scipy.io as sio
import numpy as np
from optparse import OptionParser
import time


# import gdown
# import zipfile

# CUDA_visible_devices = 1

# seed = 11
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
##torch.cuda.seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from gnn import GNN
from conv import GNN_node_Virtualnode


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)



class TripletLoss(nn.Module):
    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.l, self.r = 1, 1
        step = args.epochs // 5
        self.Ls = {
            step * 0: (0, 10),
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),
        }


    def dist(self, ins, pos):
        return torch.norm(ins - pos, dim=1)

    def forward(self, x, lens, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
                   (neg_embed_dist - neg_dist) ** 2 + \
                   (pos_neg_embed_dist - pos_neg_dist) ** 2

        return torch.mean(rank_loss), \
               torch.mean(mse_loss), \
               torch.mean(self.l * rank_loss +
                          self.r *  torch.sqrt(mse_loss))



class MLP(torch.nn.Module):
    def __init__(self, input_dim=44, num_mlp_layers=5, emb_dim=200, drop_ratio=0, multi_model=False):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.multi_model = multi_model
        #   self.emb_dim = emb_dim
        # mlp
        module_list = [
            torch.nn.Linear(input_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_ratio),
        ]

        for i in range(self.num_mlp_layers - 1):
            module_list += [torch.nn.Linear(self.emb_dim, self.emb_dim),
                            torch.nn.BatchNorm1d(self.emb_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=self.drop_ratio)]

        # relu is applied in the last layer to ensure positivity
        if not multi_model:
            module_list += [torch.nn.Linear(self.emb_dim, 1)]

        self.mlp = torch.nn.Sequential(
            *module_list
        )

    def forward(self, x):

        output = self.mlp(x)
        if self.multi_model:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)


class KDD(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, in_channels=9, emb_dim=300,
                     gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0, JK="last", graph_pooling="sum",
                     multi_model=False):
        super(KDD, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.multiple_model = multi_model
        self.in_channels = in_channels
        self.gnn_type = gnn_type
        self.virtual_node = virtual_node
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.mol_mlp = MLP(input_dim=35, num_mlp_layers=5, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.last_mlp = MLP(input_dim=self.emb_dim * 2, num_mlp_layers=10, emb_dim=self.emb_dim,
                            drop_ratio=self.drop_ratio, multi_model=False)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                   gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                   drop_ratio=self.drop_ratio, JK=self.JK,
                   graph_pooling=self.graph_pooling, multi_model=True)

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data, x):
      #  print(x)
        x_mol = torch.reshape(batched_data["mol_attr"], (-1, 35))
        # print(x_mol.shape, batched_data.spice.shape)

        #        x_mol =  torch.cat((x_mol , batched_data.spice), dim=1)
        x_mol = self.mol_mlp(x_mol)
        input = torch.cat((x, x_mol), dim=1)
        return  self.last_mlp(input)



      #  return self.graph_pred_linear(x)




class KDD_Normal(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, in_channels=9, emb_dim=300,
                     gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0, JK="last", graph_pooling="sum",
                     multi_model=False):
        super(KDD_Normal, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.multiple_model = multi_model
        self.in_channels = in_channels
        self.gnn_type = gnn_type
        self.virtual_node = virtual_node
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.mol_mlp = MLP(input_dim=35, num_mlp_layers=5, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.last_mlp = MLP(input_dim=self.emb_dim * 3, num_mlp_layers=10, emb_dim=self.emb_dim,
                            drop_ratio=self.drop_ratio, multi_model=False)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                   gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                   drop_ratio=self.drop_ratio, JK=self.JK,
                   graph_pooling=self.graph_pooling, multi_model=True)

        self.empty_subgraph = torch.from_numpy(np.array([np.repeat(0, self.emb_dim)]))

    def forward(self, batched_data):
        #print(batched_data)
        x_subs = None
        for i in range(len(batched_data.smiles)):
            x_temp = self.empty_subgraph.to(batched_data.edge_index.device)
            if x_subs is None:
                x_subs = x_temp
            else:
                x_subs = torch.cat((x_subs, x_temp), dim=0)

        x = self.gin(batched_data)
        x_mol = torch.reshape(batched_data["mol_attr"], (-1, 35))
        x_mol = self.mol_mlp(x_mol)
        input = torch.cat((x, x_mol, x_subs), dim=1)
        return self.last_mlp(input)




class KDD_Sub(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, in_channels=9, emb_dim=300,
                     gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0, JK="last", graph_pooling="sum",
                     multi_model=False):
        super(KDD_Sub, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.multiple_model = multi_model
        self.in_channels = in_channels
        self.gnn_type = gnn_type
        self.virtual_node = virtual_node
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.mol_mlp = MLP(input_dim=35, num_mlp_layers=5, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.last_mlp = MLP(input_dim=self.emb_dim * 3, num_mlp_layers=10, emb_dim=self.emb_dim ,
                            drop_ratio=self.drop_ratio, multi_model=False)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                   gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                   drop_ratio=self.drop_ratio, JK=self.JK,
                   graph_pooling=self.graph_pooling, multi_model=True)


        self.empty_subgraph = torch.from_numpy(np.array([np.repeat(0, self.emb_dim)]))

    def forward(self, batched_data, data_subs):
        #print(batched_data)
        x_subs = None
        for subs in data_subs:

            if len(subs) > 1:
                x_temp = torch.sum(self.gin(subs.to(batched_data.edge_index.device)), keepdim=True, dim=0)
            else:
                x_temp = self.empty_subgraph.to(batched_data.edge_index.device)

            if x_subs is None:
                x_subs = x_temp
            else:
                x_subs = torch.cat((x_subs, x_temp), dim = 0)

        x = self.gin(batched_data)
        x_mol = torch.reshape(batched_data["mol_attr"], (-1, 35))
        #print(x_mol.shape, batched_data.spice.shape)
        x_mol = self.mol_mlp(x_mol)
      #  print(x.shape, x_mol.shape, x_subs.shape)
        input = torch.cat(( x, x_mol, x_subs), dim=1)
        return self.last_mlp(input)




class KDD_TriAll(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, in_channels=9, emb_dim=300,
                     gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0, JK="last", graph_pooling="sum",
                     multi_model=False):
        super(KDD_TriAll, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.multiple_model = multi_model
        self.in_channels = in_channels
        self.gnn_type = gnn_type
        self.virtual_node = virtual_node
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.mol_mlp = MLP(input_dim=35, num_mlp_layers=5, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.last_mlp = MLP(input_dim=self.emb_dim * 2, num_mlp_layers=5, emb_dim=self.emb_dim,
                            drop_ratio=self.drop_ratio, multi_model=False)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                   gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                   drop_ratio=self.drop_ratio, JK=self.JK,
                   graph_pooling=self.graph_pooling, multi_model=True)



    def forward(self, batched_data, basic_batch):
        #print(batched_data)
        x_basic = self.gin(basic_batch)
        x = self.gin(batched_data)

        pos = torch.index_select(x_basic, 0, batched_data.spice_pos)
        neg = torch.index_select(x_basic, 0, batched_data.spice_neg)
        # print(batched_data["smiles"], x)
        x_mol = torch.reshape(batched_data["mol_attr"], (-1, 35))
        #print(x_mol.shape, batched_data.spice.shape)

#        x_mol =  torch.cat((x_mol , batched_data.spice), dim=1)
        x_mol = self.mol_mlp(x_mol)
        input = torch.cat((x, x_mol), dim=1)
        return x, pos, neg, self.last_mlp(input)

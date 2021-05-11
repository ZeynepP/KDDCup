import torch

from torch.utils.data.dataloader import default_collate
from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset, Batch, Data, DataLoader, dataloader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set

from conv import GNN_node, GNN_node_Virtualnode
from gnn import GNN
from main_mlpfp import MLP

import copy
from itertools import repeat, product

import torch
from torch_geometric.data import Dataset
import numpy as np
from utils.utils_mol import smiles2graph, smiles2subgraphs, get_global_features


class SubGraphModel(torch.nn.Module):

    def __init__(self, device, num_tasks=1, num_layers=3, emb_dim=300, gnn_type='gin', virtual_node=True,
                 residual=False,
                 drop_ratio=0, JK="last", graph_pooling="sum"):
        super(SubGraphModel, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.virtual_node = virtual_node
        self.residual = residual
        self.gnn_type = gnn_type
        self.mlp_dim = emb_dim
        self.device = device
        self.tempx = np.repeat(0, 9)
        self.subsmiles2graph = smiles2subgraphs
        self.empty_subgraph = torch.from_numpy(np.array([self.tempx])).to(self.device,non_blocking=True)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")
        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim=9, JK=JK, drop_ratio=drop_ratio, residual=True,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim=9, JK=JK, drop_ratio=drop_ratio, residual=True,
                                     gnn_type=gnn_type)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                       gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                       drop_ratio=self.drop_ratio, JK=self.JK,
                       graph_pooling=self.graph_pooling, multi_model=True)

        self.mlp = MLP(num_mlp_layers=self.num_layers, emb_dim=self.mlp_dim, drop_ratio=self.drop_ratio,
                       multi_model=True)

        module_list = [
            torch.nn.Linear(self.emb_dim * 2, self.mlp_dim),
            torch.nn.BatchNorm1d(self.mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_ratio),
        ]

        for i in range(self.num_layers - 1):
            module_list += [torch.nn.Linear(self.mlp_dim, self.mlp_dim),
                            torch.nn.BatchNorm1d(self.mlp_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=self.drop_ratio)]

        # relu is applied in the last layer to ensure positivity

        module_list += [torch.nn.Linear(self.mlp_dim, 1)]

        self.last = torch.nn.Sequential(
            *module_list
        )

    def forward(self, batched_data):
        # subs_features =[]
        # print(idxs)
        # for batch_temp in batched_data:
        #     if len(batch_temp) > 1:
        #         batch_temp = batch_temp.to(self.device, non_blocking=True)
        #         h_node = self.gnn_node(batch_temp)
        #         x = self.pool(h_node, batch_temp.batch)
        #         subs_features.append(torch.sum(x, keepdim=True, dim=0))
        #     else:
        #         # print(batch_temp)
        #         subs_features.append(self.empty_subgraph)
        #
        #
        # sub_input = torch.cat(subs_features, dim=0)

        y_mbtr = torch.reshape(batched_data["mol_attr"], (-1, 35))
       # print(sub_input.shape, y_mbtr.shape)
       # input = torch.cat((sub_input, y_mbtr), dim=1)

        x2 = self.mlp(y_mbtr)
        x = self.gin(batched_data)

        input = torch.cat((x, x2), dim=1)

        output = self.last(input)

        return output



    # def forward(self, batched_data):
    #     subs_features = []
    #   #  subs_features = np.array([])
    #     for subg in batched_data.subgraphs:
    #         data_graph = []
    #         for graph in subg:
    #             data_graph.append(self.get_graph_data(graph))
    #         if len(subg) > 1:
    #             batch_temp = Batch.from_data_list(data_graph).to(self.device)
    #             h_node = self.gnn_node(batch_temp)
    #             x = self.pool(h_node, batch_temp.batch)
    #             subs_features.append(torch.sum(x, keepdim=True, dim=0))
    #         else:
    #             subs_features.append(torch.from_numpy(np.array([self.tempx])).to(self.device))
    #
    #     sub_input = torch.cat(subs_features, dim=0)
    #
    #     y_mbtr = torch.reshape(batched_data["mol_attr"], (-1, 35))
    #     input = torch.cat((sub_input, y_mbtr), dim=1)
    #
    #     x2 = self.mlp(input)
    #     x = self.gin(batched_data)
    #
    #     input = torch.cat((x, x2), dim=1)
    #
    #     output = self.last(input)
    #
    #     return output


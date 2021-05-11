
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from conv import GNN_node, GNN_node_Virtualnode
from gnn import GNN
import torch
import numpy as np
from utils.utils_mol import smiles2graph, smiles2subgraphs, get_global_features


class SubGraphModel(torch.nn.Module):

    def __init__(self,  num_tasks=1, num_layers=3, emb_dim=300, gnn_type='gin', virtual_node=True,
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
      #  self.device = device

        self.subsmiles2graph = smiles2subgraphs

        # self.tempx = np.repeat(0, 9)
        # self.empty_subgraph = torch.from_numpy(np.array([self.tempx])).to(self.device,non_blocking=True)

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



    def forward(self, batched_data):
        # print("batched_data", batched_data)
        # print('sub_batched_data', sub_batched_data)
        # print(batched_data.smiles)

        if len(batched_data) > 1:
            h_node = self.gnn_node(batched_data)
            x = self.pool(h_node, batched_data.batch)
            return torch.sum(x, keepdim=True, dim=0)
        else:
            return None
            # print(batch_temp)
            #return self.empty_subgraph

#I am not able to put all to multiple gpus :(
       # return torch.cat(subs_features, dim=0)


from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set

from conv import GNN_node, GNN_node_Virtualnode
from gnn import GNN

import torch
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, input_dim=44, num_mlp_layers=5, emb_dim=300, drop_ratio=0, multi_model=False):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
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
        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)


class MultipleModel(torch.nn.Module):

    def __init__(self, num_tasks=1, num_layers=3, emb_dim=300, gnn_type='gin', virtual_node=True,
                 residual=False,
                 drop_ratio=0, JK="last", graph_pooling="sum"):
        super(MultipleModel, self).__init__()

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

        self.mlp = MLP(input_dim=44, num_mlp_layers=self.num_layers, emb_dim=self.mlp_dim, drop_ratio=self.drop_ratio,
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

        self.last_mlp = torch.nn.Sequential(
            *module_list
        )


    def forward(self, batched_data, sub_input):

        y_mbtr = torch.reshape(batched_data["mol_attr"], (-1, 35))
       # print(sub_input.shape, y_mbtr.shape)
        input = torch.cat((sub_input, y_mbtr), dim=1)

        x2 = self.mlp(input)
        x = self.gin(batched_data)

        input = torch.cat((x, x2), dim=1)

        output = self.last_mlp(input)

        return output


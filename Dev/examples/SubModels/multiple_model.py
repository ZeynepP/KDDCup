from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, SAGEConv

from conv import GNN_node, GNN_node_Virtualnode
from gnn import GNN
import torch.nn.functional as F
import torch
import numpy as np


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


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


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


        self.sage = SAGE(in_channels=9, hidden_channels=self.emb_dim, num_layers=self.num_layers)

        self.gin = GNN(num_tasks=self.num_tasks, num_layers=self.num_layers, emb_dim=self.emb_dim,
                       gnn_type=self.gnn_type, virtual_node=self.virtual_node, residual=self.residual,
                       drop_ratio=self.drop_ratio, JK=self.JK,
                       graph_pooling=self.graph_pooling, multi_model=True)

        self.mol_mlp = MLP(input_dim=35, num_mlp_layers=self.num_layers, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.sub_mlp = MLP(input_dim=self.emb_dim, num_mlp_layers=self.num_layers, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=True)

        self.last_mlp = MLP(input_dim=self.emb_dim * 3, num_mlp_layers=self.num_layers, emb_dim=self.emb_dim,
                           drop_ratio=self.drop_ratio, multi_model=False)


        self.tempx = np.repeat(0, 200)
        self.empty_subgraph = torch.from_numpy(np.array([self.tempx]))

    def forward(self, batched_data, sub_batched_data):
        # print("batched_data", batched_data)
        # print('sub_batched_data', sub_batched_data)
        # print(batched_data.smiles)
        subs_features = []
        device = batched_data.edge_index.device
        for batch_temp in sub_batched_data:

            if len(batch_temp) > 1:
                batch_temp = batch_temp.to(device)
                h_node = self.sage(batch_temp.x.type(torch.FloatTensor).to(device), batch_temp.edge_index.type(torch.LongTensor).to(device))
                x = self.pool(h_node, batch_temp.batch)
                subs_features.append(torch.sum(x, keepdim=True, dim=0))
            else:
                # print(batch_temp)
                subs_features.append(self.empty_subgraph.to(batched_data.edge_index.device))

        x_sub = torch.cat(subs_features, dim=0)
        x_sub = self.sub_mlp(x_sub)

        x_mol = self.mol_mlp(torch.reshape(batched_data["mol_attr"], (-1, 35)))

        x = self.gin(batched_data)

        input = torch.cat((x, x_mol, x_sub), dim=1)
        return self.last_mlp(input)



##############" MODEL 1 ############################
# def forward(self, batched_data, sub_batched_data):
#     # print("batched_data", batched_data)
#     # print('sub_batched_data', sub_batched_data)
#     # print(batched_data.smiles)
#     subs_features =[]
#     for batch_temp in sub_batched_data:
#
#         if len(batch_temp) > 1:
#             batch_temp = batch_temp.to(batched_data.edge_index.device)
#             h_node = self.gnn_node(batch_temp)
#             x = self.pool(h_node, batch_temp.batch)
#             subs_features.append(torch.sum(x, keepdim=True, dim=0))
#         else:
#             # print(batch_temp)
#             subs_features.append(self.empty_subgraph.to(batched_data.edge_index.device))
#
#
#     sub_input = torch.cat(subs_features, dim=0)
#
#     y_mbtr = torch.reshape(batched_data["mol_attr"], (-1, 35))
#
#     input = torch.cat((sub_input, y_mbtr), dim=1)
#
#     x2 = self.mlp(input)
#     x = self.gin(batched_data)
#
#     input = torch.cat((x, x2), dim=1)
#
#     output = self.last_mlp(input)
#
#     return output

##############" MODEL 2 ############################
# def forward(self, batched_data, sub_batched_data):
#     # print("batched_data", batched_data)
#     # print('sub_batched_data', sub_batched_data)
#     # print(batched_data.smiles)
#     subs_features = []
#     for batch_temp in sub_batched_data:
#
#         if len(batch_temp) > 1:
#             batch_temp = batch_temp.to(batched_data.edge_index.device)
#             h_node = self.gnn_node(batch_temp)
#             x = self.pool(h_node, batch_temp.batch)
#             subs_features.append(torch.sum(x, keepdim=True, dim=0))
#         else:
#             # print(batch_temp)
#             subs_features.append(self.empty_subgraph.to(batched_data.edge_index.device))
#
#     sub_input = torch.cat(subs_features, dim=0)
#
#     y_mbtr = torch.reshape(batched_data["mol_attr"], (-1, 35))
#     x = self.gin(batched_data)
#     input = torch.cat((x, sub_input, y_mbtr), dim=1)
#
#     output = self.last_mlp(input)
#
#     return output
#

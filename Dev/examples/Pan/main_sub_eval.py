import io

import torch
import torch.optim as optim
from ogb.lsc import PCQM4MEvaluator
from sklearn.model_selection import ShuffleSplit

from torch_geometric.data import DataLoader, Batch, Data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from utils.utils_mol import smiles2graph
from utils.pcqm4m_pyg import PygPCQM4MDataset

import os.path as osp
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random
import pandas as pd
from triplet_model import KDD_Normal, KDD_Sub

reg_criterion = torch.nn.L1Loss()
MIN_IDX = 1764198
MAX_IDX = 2919966
MEAN_IDX = 1701147
suball = pd.read_csv("/usr/src/kdd/suball.csv")
cache ={}
def get_dataloaders(args):
    ### automatic dataloading and splitting
    dataset = PygPCQM4MDataset(root=args.data_dir)
    print(dataset.len())
    split_idx = dataset.get_idx_split()

    # # ########################## TEST ##################################
    # split_idx = {}
    # # this part for test
    # ids = [i for i in range(100)]
    #
    # rs = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    # for train_index, test_index in rs.split(ids, ids):
    #     split_idx["train"], split_idx["test"] = train_index, test_index
    #
    # rs = ShuffleSplit(n_splits=1, test_size=.5, random_state=0)
    # for train_index, test_index in rs.split(split_idx["test"], split_idx["test"]):
    #     split_idx["valid"], split_idx["test"] = train_index, test_index
    #
    # split_idx["train"] = torch.from_numpy(np.array(split_idx["train"], dtype=np.int64))
    # split_idx["test"] = torch.from_numpy(np.array(split_idx["test"], dtype=np.int64))
    # split_idx["valid"] = torch.from_numpy(np.array(split_idx["valid"], dtype=np.int64))
    # # #############################################################"

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = None
    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    return train_loader, valid_loader, test_loader



def read_subgraphs(idx):
        try:
            with open('/data/processed/geometric_data_processed_data_{}.pt'.format(idx), 'rb') as f:
                sub = torch.load(io.BytesIO(f.read()))
            # sub = torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx))) too many open files
        except Exception as ex:
            sub = []

        return sub


def get_data( graph):
    data = Data()
    if graph is not None:
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        data.nn = int(graph['nodes_num'])
        data.smiles = str(graph["smiles"])  # to check
    return data

def get_data_subgraph( idx):

    if idx in cache:
        return cache[idx]
    else:
        subgraphs = suball[(suball["idx"] == idx) & (suball.sub_smiles.str.len() > 10)].sub_smiles.values
        data_list = []
        if len(subgraphs) > 1:
            for sub in subgraphs:
                graph = smiles2graph(sub)
                data_list.append(get_data(graph))
            cache[idx] = Batch.from_data_list(data_list)
        else:
            cache[idx] = data_list
        return cache[idx]

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        subs =[]
        for idx in batch.idx:
            subs.append(read_subgraphs(idx))

        pred = model(batch.to(device), subs).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        subs = []
        for idx in batch.idx:
            subs.append(read_subgraphs(idx))

        with torch.no_grad():
            pred = model(batch.to(device), subs).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return float(evaluator.eval(input_dict)["mae"])




def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        subs = []
        for idx in batch.idx:
            subs.append(get_data_subgraph(idx))

        with torch.no_grad():
            pred = model(batch.to(device), subs).view(-1, )

    y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred




def main(args):
    print(args)
    num_node_features = 9
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_name = 'checkpoint_sub.pt'
    print(device, args.data_dir)
    shared_params = {
        "in_channels": num_node_features,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }
    # model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    model = KDD_Sub(**shared_params).to(device)


    train_loader, valid_loader, test_loader = get_dataloaders(args)
    evaluator = PCQM4MEvaluator()
#    optimizer = optim.Adam(model.parameters(), lr=0.001)
  #  scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
 #   scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, verbose=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_sub.pt')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f'Checkpoint file not found at {checkpoint_path}')

    ## reading in checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Model Params: {num_params}')

    valid_mae = eval(model, device, valid_loader, evaluator)

    print({'Validation': valid_mae})



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=7,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default="/usr/src/kdd/data_test6/",
                        help='data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')


    args = parser.parse_args()


    main(args)






import torch


import torch.optim as optim
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch

from multiple_model import MultipleModel
from utils.subgraphs_dataset import SubGraphsPCQM4MDataset
from utils.sub_dataset_2 import SubPygPCQM4MDataset
from subgraph_model import SubGraphModel

import os.path as osp
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random
import cProfile, pstats, io
from pstats import SortKey
### importing OGB-LSC
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator, PCQM4MDataset

reg_criterion = torch.nn.L1Loss()
torch.multiprocessing.set_sharing_strategy('file_system')


def sub_collate(a):

    data =[]
    subs=[]
    for b in a:
        data.append(b[0])
        subs.append(b[1])
    return Batch.from_data_list(data), subs



def train(model, sub_model, device, loader, optimizer, sub_batches):

   # dev = f'cuda:{model.device_ids[0]}'
   # sub_dev = f'cuda:{sub_model.device_ids[0]}'
  #  print(dev, sub_dev)



    tempx = np.repeat(0, 9)
    empty_subgraph = torch.from_numpy(np.array([tempx])).to(device)
    sub_model.train()
    model.train()
    loss_accum = 0

    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub = sub_batches.get(idx)
            if len(sub) > 0 :
                sub = sub.to(device)
                sub_pred = sub_model(sub)
            else:
                sub_pred = empty_subgraph
            sub_features.append(sub_pred)

        sub_input = torch.cat(sub_features, dim=0).to(device)

        pred = model(batch,sub_input).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, sub_model, device, loader, evaluator,sub_batches):
    model.eval()
    y_true = []
    y_pred = []
    tempx = np.repeat(0, 9)
    empty_subgraph = torch.from_numpy(np.array([tempx])).to(device)
    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub = sub_batches.get(idx)
            if len(sub) > 0:
                sub = sub.to(device)
                sub_pred = sub_model(sub)
            else:
                sub_pred = empty_subgraph
            sub_features.append(sub_pred)

        sub_input = torch.cat(sub_features, dim=0).to(device)

        with torch.no_grad():
            pred = model(batch, sub_input).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]




def test(model, sub_model, device, loader, sub_batches):

    model.eval()
    y_pred = []


    tempx = np.repeat(0, 9)
    empty_subgraph = torch.from_numpy(np.array([tempx])).to(device)
    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub = sub_batches.get(idx)
            if len(sub) > 0:
                sub = sub.to(device)
                sub_pred = sub_model(sub)
            else:
                sub_pred = empty_subgraph
            sub_features.append(sub_pred)

        sub_input = torch.cat(sub_features, dim=0).to(device)

        with torch.no_grad():
            pred = model(batch, sub_input).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


import multiprocessing as mp
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default="/home/zpehlivan/Ina/Data/KDD/",
                        help='data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    #tryin to use multiple gpus
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print(device, args.data_dir)

    #dataloading for sub graphs
    subdataset = SubGraphsPCQM4MDataset(root='/data/processed/')
    print(len(subdataset.batches))

    ### automatic dataloading and splitting
    dataset = SubPygPCQM4MDataset(root=args.data_dir)
    print(dataset.len())
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()
    split_idx = {}
    ids = [i for i in range(100000)]

    rs = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    for train_index, test_index in rs.split(ids, ids):
        split_idx["train"], split_idx["test"] = train_index, test_index

    rs = ShuffleSplit(n_splits=1, test_size=.5, random_state=0)
    for train_index, test_index in rs.split(split_idx["test"], split_idx["test"]):
        split_idx["valid"], split_idx["test"] = train_index, test_index

    split_idx["train"] = torch.from_numpy(np.array(split_idx["train"], dtype=np.int64))
    split_idx["test"] =  torch.from_numpy(np.array(split_idx["test"], dtype=np.int64))
    split_idx["valid"] =  torch.from_numpy(np.array(split_idx["valid"], dtype=np.int64))

    #args.train_subset = True
    if args.train_subset:
        print("Training subset ...")
        subset_ratio = 0.01
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]

        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=sub_collate)


    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=sub_collate)

    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=sub_collate)

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        sub_model = SubGraphModel(gnn_type='gin', virtual_node=False,  **shared_params)
        model = MultipleModel(gnn_type='gin', virtual_node=False,  **shared_params)
    elif args.gnn == 'gin-virtual':
        sub_model = SubGraphModel(gnn_type='gin', virtual_node=True, **shared_params)
        model = MultipleModel(gnn_type='gin', virtual_node=True, **shared_params)
    elif args.gnn == 'gcn':
        sub_model = SubGraphModel(gnn_type='gcn', virtual_node=False, **shared_params)
        model = MultipleModel(gnn_type='gcn', virtual_node=False,  **shared_params)
    elif args.gnn == 'gcn-virtual':
        sub_model = SubGraphModel(gnn_type='gcn', virtual_node=True, **shared_params)
        model = MultipleModel(gnn_type='gcn', virtual_node=True,  **shared_params)
    else:
        raise ValueError('Invalid GNN type')

    sub_model.to(device)
    model.to(device)
  #  sub_model = torch.nn.DataParallel(sub_model)
  #  model = torch.nn.DataParallel(model)

    num_params = sum(p.numel() for p in sub_model.parameters())
    print(f'#Sub Model Params: {num_params}')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Model Params: {num_params}')


    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        args.epochs = 100
    else:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        # pr = cProfile.Profile()
        # pr.enable()
        # # ... do something ...
        #

        train_mae = train(model, sub_model, device, train_loader,  optimizer, subdataset.batches)

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        print('Evaluating...')

        valid_mae = eval(model, sub_model, device, valid_loader, evaluator,subdataset.batches)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir is not '':
                print('Predicting on test data...')
                y_pred = test(model, sub_model, device, test_loader, subdataset.batches)
                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir is not '':
        writer.close()


if __name__ == "__main__":
    main()
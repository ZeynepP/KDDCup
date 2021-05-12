import torch

import torch.optim as optim
from ogb.lsc import PCQM4MEvaluator
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, ConcatDataset

from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.data import Batch

from sage import SAGE
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
reg_criterion = torch.nn.L1Loss()


def sub_collate(a):
    data = []
    subs = []
    for b in a:
        data.append(b[0])
        subs.append(b[1])
    return Batch.from_data_list(data), subs


def get_dataloaders(args):
    ### automatic dataloading and splitting
    dataset = SubPygPCQM4MDataset(root=args.data_dir)
    print(dataset.len())
    split_idx = dataset.get_idx_split()

    ########################### TEST ##################################
    # split_idx = {}
    # # this part for test
    # ids = [i for i in range(100000)]
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
    #############################################################"

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=sub_collate)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=sub_collate)
    test_loader = None
    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=sub_collate)

    return train_loader, valid_loader, test_loader


def get_model(args):
    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':

        model = MultipleModel(gnn_type='gin', virtual_node=False, **shared_params)
    elif args.gnn == 'gin-virtual':
      #  sub_model = SubGraphModel(gnn_type='gin', virtual_node=True, **shared_params)
        model = MultipleModel(gnn_type='gin', virtual_node=True, **shared_params)
    elif args.gnn == 'gcn':
       # sub_model = SubGraphModel(gnn_type='gcn', virtual_node=False, **shared_params)
        model = MultipleModel(gnn_type='gcn', virtual_node=False, **shared_params)
    elif args.gnn == 'gcn-virtual':
        #sub_model = SubGraphModel(gnn_type='gcn', virtual_node=True, **shared_params)
        model = MultipleModel(gnn_type='gcn', virtual_node=True, **shared_params)
    else:
        raise ValueError('Invalid GNN type')

    return model, SAGE(9, hidden_channels=64, num_layers=2)


def train(model, device, loader, optimizer, sub_batches):
    model.train()
    loss_accum = 0

    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub_features.append(sub_batches.get(idx))

        pred = model(batch, sub_features).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, sub_batches):
    model.eval()
    y_true = []
    y_pred = []

    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub_features.append(sub_batches.get(idx))

        with torch.no_grad():
            pred = model(batch, sub_features).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader, sub_batches):
    model.eval()
    y_pred = []

    for step, batcht in enumerate(tqdm(loader, desc="Iteration")):

        batch, subs = batcht
        batch = batch.to(device)

        sub_features = []
        for idx in subs:
            sub_features.append(sub_batches.get(idx))

        with torch.no_grad():
            pred = model(batch, sub_features).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred

def main(args, subdataset):
    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print(device, args.data_dir)
    model, sub_model = get_model(args)
    train_loader, valid_loader, test_loader = get_dataloaders(args)

    model.to(device)
    sub_model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Model Params: {num_params}')

    evaluator = PCQM4MEvaluator()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
  #  scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.02)

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer, subdataset.batches)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator, subdataset.batches)

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
                y_pred = test(model, device, test_loader, subdataset.batches)
                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir is not '':
        writer.close()


if __name__ == "__main__":
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
    subdataset = SubGraphsPCQM4MDataset(root='/data/processed/')
    print(len(subdataset.batches))
    main(args,subdataset)





import io

import torch

import torch.optim as optim
from ogb.lsc import PCQM4MEvaluator
from sklearn.model_selection import ShuffleSplit
#from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import PairwiseDistance, CosineSimilarity

from torch_geometric.data import DataLoader, Batch

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


import os

from torch_geometric.nn import DataParallel
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from triplet_utils import AllTripletSelector, SemihardNegativeTripletSelector, HardestNegativeTripletSelector
from loss import SCTLoss
from utils.pcqm4m_pyg import PygPCQM4MDataset
from gnn import GNN
from triplet_model import KDD
import networkx as nx
reg_criterion = torch.nn.L1Loss()
#triplet_criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=CosineSimilarity())
import os.path as osp


class OnlineTestTriplet(torch.nn.Module):
    def __init__(self, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        return self.triplet_selector.get_triplets(embeddings, target)


class TripletsTrain(torch.nn.Module):

    def __init__(self, device, loader, epochs ,args):
        super(TripletsTrain, self).__init__()


        self.trip_loss_fun = torch.nn.TripletMarginLoss()
        self.device = device
        self.checkpoint_file = args.checkpoint_dir
        self.triplet_selector = SemihardNegativeTripletSelector(margin=1) #AllTripletSelector() #
        self.TripSel = OnlineTestTriplet( self.triplet_selector).to(self.device)
        self.loader = loader
        self.epochs  = 3

        self.model =  GNN(num_tasks=1, num_layers=args.num_layers, emb_dim=args.emb_dim,
                       gnn_type='gin', virtual_node=True, drop_ratio=args.drop_ratio,
                       graph_pooling=args.graph_pooling, multi_model=True).to(self.device)

        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f'#Model Emb Params: {self.num_params}')

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=3, verbose=True)



    def train(self):
        self.model.train()
        loss_accum = 0
        for step, batch in enumerate(tqdm(self.loader, desc="Iteration")):
            batch = batch.to(self.device)
            encoded = self.model(batch)
            triplets_list = self.TripSel(encoded,  batch.labels)

            t_loss = self.trip_loss_fun(encoded[triplets_list[:, 0], :], encoded[triplets_list[:, 1], :],
                                   encoded[triplets_list[:, 2], :])

            self.optimizer.zero_grad()
            t_loss.backward()
            self.optimizer.step()

        return loss_accum / (step + 1)

    def run(self):
        best_loss = 1000
        for epoch in range(1, self.epochs + 1):
            print("=====Epoch Emb{}".format(epoch))
            print('Training Emb...')
            train_loss = self.train()
            print({'Train': train_loss})
            if train_loss < best_loss:
                best_loss = train_loss
                if args.checkpoint_dir is not '':
                    print('Emb Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                  'optimizer_state_dict': self.optimizer.state_dict(),
                                  'scheduler_state_dict': self.scheduler.state_dict(), 'best_val_mae': best_loss,
                                  'num_params': self.num_params}

                    torch.save(checkpoint, os.path.join(self.checkpoint_file, 'checkpoint_emb_hard.pt'))

            self.scheduler.step(train_loss)
            if epoch == 1:
                print("Updating triplet selector...")
                self.triplet_selector = HardestNegativeTripletSelector(margin=1)  # AllTripletSelector() #
                self.TripSel = OnlineTestTriplet(self.triplet_selector).to(self.device)

            print(f'Emb best validation MAE so far: {train_loss}')
        print("Over training emb...")
        return self.model

def get_dataloaders(args):
    ### automatic dataloading and splitting
    dataset = PygPCQM4MDataset(root=args.data_dir)
    print(dataset.len())
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = None
    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    return train_loader, valid_loader, test_loader


def train(model, device, loader, optimizer, model_emb):
        model.train()
        loss_accum = 0

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            a = model_emb(batch.to(device))
            pred = model(batch.to(device), a)

            loss = reg_criterion(pred.view(-1, ), batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += float(loss.detach().cpu().item())

        return loss_accum / (step + 1)


def eval(model, device, loader, evaluator,  model_emb):
        model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            with torch.no_grad():
                a = model_emb(batch.to(device))
                pred = model(batch.to(device), a)
                pred = pred.view(-1, )
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return float( evaluator.eval(input_dict)["mae"])

def test(model, device, loader, model_emb):
        model.eval()
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            with torch.no_grad():
                a = model_emb(batch.to(device))
                pred = model(batch.to(device), a)
                pred = pred.view(-1, )

            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)

        return y_pred


def main(args):
    print(args)

    skip_emb = True

    num_node_features = 9
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print(device, args.data_dir)

    train_loader, valid_loader, test_loader = get_dataloaders(args)
    evaluator = PCQM4MEvaluator()



#device, loader, epochs , checkpoint_file, margin, model
    if skip_emb:
        model_emb =  GNN(num_tasks=1, num_layers=args.num_layers, emb_dim=args.emb_dim,
                       gnn_type='gin', virtual_node=True, drop_ratio=args.drop_ratio,
                       graph_pooling=args.graph_pooling, multi_model=True)

        checkpoint = torch.load( os.path.join(args.checkpoint_dir, 'checkpoint_emb_hard.pt'))
        print(checkpoint['model_state_dict'])
        model_emb.load_state_dict(checkpoint['model_state_dict'])
        model_emb = model_emb.to(device)
    else:
        tt  = TripletsTrain(device = device, loader=train_loader, epochs=100, args = args )
        model_emb= tt.run()

    shared_params = {
        "in_channels": num_node_features,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }
    model_mlp = KDD(**shared_params).to(device)
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.003651)
    scheduler_mlp = ReduceLROnPlateau(optimizer_mlp, 'min', factor=0.2, patience=3, verbose=True)

    num_params = sum(p.numel() for p in model_mlp.parameters())
    print(f'#Model Params: {num_params}')

    if args.log_dir is not '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model_mlp, device, train_loader, optimizer_mlp, model_emb)

        print('Evaluating...')
        valid_mae = eval(model_mlp, device, valid_loader, evaluator, model_emb)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir is not '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model_mlp.state_dict(),
                              'optimizer_state_dict': optimizer_mlp.state_dict(),
                              'scheduler_state_dict': scheduler_mlp.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}

                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_mlp_triplet.pt'))

        scheduler_mlp.step(valid_mae)

        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir is not '':
        writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=200,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=48,
                        help='number of workers (default: 0)')
    parser.add_argument('--margin', type=int, default=1,
                        help='margin (default: 0.1)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default="/usr/src/kdd/data_test6/",
                        help='data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='/usr/src/kdd/checkpoint_gin_sub/', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')

    args = parser.parse_args()

    main(args)





import io
import multiprocessing as mp
import os
import gc
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch
#os.system("taskset -p 0xff %d" % os.getpid())
from torch_geometric.data import InMemoryDataset, dataloader, Batch
from torch_geometric.data import Data
import copy
from itertools import repeat, product
from utils.utils_mol import smiles2graph, smiles2subgraphs, get_global_features


class SubGraphsPCQM4MDataset():


    def __init__(self, root='dataset', smiles2subgraphs=smiles2subgraphs,  transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4M dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.smiles2subgraphs = smiles2subgraphs

        self.extra =root


     #   super(SubGraphsPCQM4MDataset, self).__init__(self.folder, transform, pre_transform)
        print("Loading sub graphs")

        self.batches = {}

        for i in tqdm(range(100000)):#3803453
            d = self.read_subgraphs(i)
            self.batches.update(d)
        # data_list = self.load_subgraphs()
        # while len(data_list)>0:
        #     d = data_list.pop()
        #     self.batches.update(d)
        # print(len(self.batches))

# getting too many files open error
    # I increased ulimit but again
    def load_subgraphs(self):
        poolp = mp.pool.Pool(processes=10)
        print('Loading sub batches from files...')
        data_list = list(tqdm(poolp.imap_unordered(self.read_subgraphs, range(3803453)))) #
        poolp.close()
        poolp.join()

        return data_list

    def read_subgraphs(self, idx):

        try:
            with open(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx)), 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    sub = torch.load(buffer)

            #sub = torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx)))
        except Exception as ex:
            sub = []
        return {idx:sub}


    def get_data(self, graph):
        data = Data()
        if graph is not None:
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.nn = int(graph['nodes_num'])
            data.smiles = str(graph["smiles"])  # to check
        return data

    def get_data_subgraph(self,i, smiles):

        subgraphs = self.smiles2subgraphs(smiles)
        if len(subgraphs)>1:
            data_list = []
            for sub in subgraphs:
                data_list.append(self.get_data(sub))

            # print('Saving...', self.processed_paths[0] + 'data_{}.pt'.format(i))
            torch.save(Batch.from_data_list(data_list),osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(i)))



    # def process(self):
    #     data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
    #     args = list(zip(data_df.idx, data_df.smiles))
    #
    #     poolp = ProcessPool()  # multiprocessing.pool.ThreadPool(200)
    #     print('Converting SMILES strings into graphs...')
    #     data_list = list(tqdm(poolp.imap(self.get_data_subgraph, args)))
    #     poolp.close()
    #     poolp.join()
    #
    #



if __name__ == '__main__':
    dataset = SubGraphsPCQM4MDataset(root="/usr/src/kdd/data_test6/")
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[50])
    print(dataset[50].y)
    print(dataset.get_idx_split())


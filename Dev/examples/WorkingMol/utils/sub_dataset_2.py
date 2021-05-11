import multiprocessing as mp
import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch
#os.system("taskset -p 0xff %d" % os.getpid())
from pathos.multiprocessing import ProcessPool
from torch_geometric.data import InMemoryDataset, dataloader, Batch
from torch_geometric.data import Data
import copy
from itertools import repeat, product
from utils.utils_mol import smiles2graph, smiles2subgraphs, get_global_features



class SubPygPCQM4MDataset(InMemoryDataset):


    def __init__(self, root='dataset', smiles2graph=smiles2graph, smiles2subgraphs=smiles2subgraphs,
                 transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4M dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.data, self.slices = [], {}
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.smiles2subgraphs = smiles2subgraphs
        self.root = osp.join(root, 'pcqm4m_kddcup2021')
        self.folder = osp.join(root, 'pcqm4m_kddcup2021')
        self.version = 1
        self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m_kddcup2021.zip'
        self.extra =osp.join('/data/', 'processed/')

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4M dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(SubPygPCQM4MDataset, self).__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.nosubgraph_counter = 0
        self.cache={}



    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'


    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def read_subgraphs(self, idx):

            try:
                sub =torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx)))

                self.cache[idx] = sub
            except OSError:
                # we may be using too much memory
                del self.cache[list(self.cache.keys())[0]]
                try:
                    sub = torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx)))
                    self.cache[idx] = sub
                except:
                    sub = []
                    self.cache[idx] = sub

            except FileNotFoundError:
                sub = []
                self.cache[idx] = sub
                self.nosubgraph_counter += 1

            return sub
    #
    #
    # def _get_(self, idx):
    #     if hasattr(self, '__data_list__'):
    #         if self.__data_list__ is None:
    #             self.__data_list__ = self.len() * [None]
    #         else:
    #             data = self.__data_list__[idx]
    #             if data is not None:
    #                 return copy.copy(data)
    #
    #     data = self.data.__class__()
    #     if hasattr(self.data, '__num_nodes__'):
    #         data.num_nodes = self.data.__num_nodes__[idx]
    #
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         start, end = slices[idx].item(), slices[idx + 1].item()
    #         if torch.is_tensor(item):
    #             s = list(repeat(slice(None), item.dim()))
    #             s[self.data.__cat_dim__(key, item)] = slice(start, end)
    #         elif start + 1 == end:
    #             s = slices[start]
    #         else:
    #             s = slice(start, end)
    #         data[key] = item[s]
    #
    #     if hasattr(self, '__data_list__'):
    #         self.__data_list__[idx] = copy.copy(data)
    #
    #     return data
    #
    # def get(self, idx):
    #     data = self._get_(idx)
    #     # sub = self.cache.get(idx, None)
    #     # if sub is None:
    #     #     sub = self.read_subgraphs(idx)
    #     return (data, idx)
    #     # return (data,sub)
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



    def get_data_graph(self, a ):
        i, smiles, homolumogap = a
        graph = self.smiles2graph(smiles)

        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['nodes_num'])

        data = self.get_data(graph)
        data.idx = int(i)
        data.__num_nodes__ = int(graph['nodes_num'])
        data.y = torch.Tensor([homolumogap])
        # Extra feautres : Getting mol data
        data.mol_attr = torch.from_numpy(get_global_features(smiles)).to(torch.float)
     #   data.subgraphs = self.smiles2subgraphs(data.smiles)
     #   data.nsub = len(data.subgraphs)
     #   self.get_data_subgraph( i, smiles)
        return data



    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        args = list(zip(data_df.idx, data_df.smiles, data_df.homolumogap))

        poolp = ProcessPool()  # multiprocessing.pool.ThreadPool(200)
        print('Converting SMILES strings into graphs...')
        data_list = list(tqdm(poolp.imap(self.get_data_graph, args)))
        poolp.close()
        poolp.join()

        split_dict = self.get_idx_split()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        print('Saving...', self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])




    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

if __name__ == '__main__':
    dataset = SubPygPCQM4MDataset(root="/usr/src/kdd/data_test7/")
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[50])
    print(dataset[50].y)



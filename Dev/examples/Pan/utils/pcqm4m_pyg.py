import copy
import io
import os
import os.path as osp
import shutil
from itertools import repeat


from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.data import Data
from utils.utils_mol import smiles2graph

class PygPCQM4MDataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4M dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m_kddcup2021')
        self.version = 1
        self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m_kddcup2021.zip'
        self.extra = '/data/processed/'
        # self._use_smiles = False

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4M dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4MDataset, self).__init__(self.folder, transform, pre_transform)

        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        self.suball = pd.read_csv("/usr/src/kdd/suball.csv")

        #
        # MIN_DICT = ['C\\C=C\\N', 'C\\C=C/N', 'C-C(=O)-C-C=C', 'C-C(=O)-C=O',
        #             'O=C-C=C-C=O', 'C-N=C-C=C', 'C-C=C(-C)-C=C', 'C-C(-C)=C-C=C',
        #             'C=C-C(=O)-C=C', 'C=C-C=C-C=C', 'N=O', 'C-C-C(=O)-C=C',
        #             'C-C=C-C=C-N', 'C-C(=C)-C=C', 'C-C(=C)-N', 'C-C(=O)-C=C',
        #             'C=C-C=C-C=O', 'C=C-C=C-N', 'C-C(-C)=C', 'C-C=C-C=C']
        #
        #
        # MAX_DICT = ['C-N(-C)-C-C-N', 'C-O-C(-C)-C', 'C-C-C(-C)-C-O', 'C-C-C(-N)-C-C',
        #             'C1-C-C-C-C-C-1', 'C-C-C(-C)-C-N', 'C-C-C-O-C-C', 'C-N-C-C(-C)-C',
        #             'C-C-C-C-O-C', 'C-C-C-C(-C)-N', 'C-C-N-C-C-O', 'C-C-C(-C)-N-C',
        #             'C-C-N-C(-C)-C', 'C-C(-C)-C-O', 'C-C-C-N(-C)-C', 'C-C-C(-C)-C-C',
        #             'C-C-C-C-C-O', 'C-C-C-C(-C)-C', 'C-C-C-C-N-C', 'C-C-C-N-C-C']
        # #oxido, nitrogene,pyrrolidine, oxotane
        # MAIN_DICT = MAX_DICT + MIN_DICT
        #
        # self.spice = pd.DataFrame(dict((mol, suball[suball.sub_smiles==mol].groupby(suball.main_smiles).count().idx) for mol in MAIN_DICT))
        # m = pd.merge(data_df, self.spice, how="left", right_index=True, left_on="smiles")
        # print(m.head(5))
        # m.drop(columns = ["idx","smiles", "homolumogap"], inplace=True)
        # m.fillna(0, inplace=True)
        # self.spice = m.values


        data_df["spice"] = 0
        #calculated by quantiles
        data_df.loc[(data_df['homolumogap'] > 4.315726) & (data_df['homolumogap'] <= 7.257276), 'spice'] = 1
        # data_df.loc[(data_df['homolumogap'] > 4.315726) & (data_df['homolumogap'] <=  5.053154), 'spice'] = 1
        # data_df.loc[(data_df['homolumogap'] >  5.053154) & (data_df['homolumogap'] <= 5.839563), 'spice'] = 2
        # data_df.loc[(data_df['homolumogap'] > 5.839563) & (data_df['homolumogap'] <= 7.257276), 'spice'] = 3
        data_df.loc[(data_df['homolumogap'] > 7.257276) , 'spice'] = 4
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.target = data_df["spice"].values
        del data_df
        # del suball
        # del m


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
    def _get_(self, idx):
        if hasattr(self, '__data_list__'):
            if self.__data_list__ is None:
                self.__data_list__ = self.len() * [None]
            else:
                data = self.__data_list__[idx]
                if data is not None:
                    return copy.copy(data)

        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        if hasattr(self, '__data_list__'):
            self.__data_list__[idx] = copy.copy(data)

        return data

    def get(self, idx):
        data = self._get_(idx)
       # data.spice_pos = torch.from_numpy(np.array(self.spice_pos[idx])).to(torch.int64)
       # data.spice_neg = torch.from_numpy(np.array(self.spice_neg[idx])).to(torch.int64)

        data.labels = torch.from_numpy(np.array(self.target[idx])).to(torch.int64)
        data.idx = idx
    #    data.spice = torch.from_numpy(np.array([self.spice[idx]])).to(torch.int64)
        return data

    def get_data(self, graph):
        data = Data()
        if graph is not None:
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.nn = int(graph['nodes_num'])
            data.smiles = str(graph["smiles"])  # to check
        return data

    def read_subgraphs(self, idx):

        try:
            with open(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx)), 'rb') as f:
                sub = torch.load(io.BytesIO(f.read()))

            # sub = torch.load(osp.join(self.extra, 'geometric_data_processed_data_{}.pt'.format(idx))) too many open files
        except Exception as ex:
            sub = []

        return sub
    def get_data_subgraph(self, idx):

        subgraphs = self.suball[self.suball["idx"] == idx].sub_smiles.values
        data_list = []
        if len(subgraphs) > 1:

            for sub in subgraphs:
                graph = self.smiles2graph(sub)
                data_list.append(self.get_data(graph))
            return Batch.from_data_list(data_list)
        return data_list

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])
            print(graph)
            data.__num_nodes__ = int(graph['num_nodes'])
            data.neighbors = torch.from_numpy(graph['neighbors']).to(torch.int64)
            data.edge_list = torch.from_numpy(graph['edge_list']).to(torch.int64)
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.mol_attr = torch.from_numpy(graph['mol_feat']).to(torch.int64)

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


if __name__ == '__main__':
    dataset = PygPCQM4MDataset(root="/usr/src/kdd/data_test6/")
    # print(dataset.data.edge_index)
    # print(dataset.data.edge_index.shape)
    # print(dataset.data.x.shape)
    # print(dataset[100])
    # print(dataset[100].y)
    # print(dataset.get_idx_split())
    print(dataset[1].num_edges)
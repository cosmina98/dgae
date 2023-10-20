import os
import torch
import numpy as np
import pickle
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx
from utils.func import atom_number_to_one_hot, from_dense_numpy_to_sparse


class KekulizedMolDataset(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc250k_kekulized.npz']
        elif self.dataset == 'qm9':
            return ['qm9_kekulized.npz']
        else:
            raise NotImplementedError()

    @property
    def processed_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc_data.pt']
        elif self.dataset == 'qm9':
            return ['data_qm9.pt']

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'zinc':
            download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                         filename='zinc250k_kekulized.npz')
        elif self.dataset == 'qm9':
            download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                         filename='qm9_kekulized.npz')


    def process(self):
        if self.dataset == 'zinc':
            filepath = os.path.join(self.raw_dir, 'zinc250k_kekulized.npz')
        elif self.dataset == 'qm9':
            filepath = os.path.join(self.raw_dir, 'qm9_kekulized.npz')

        load_data = np.load(filepath)
        xs = load_data['arr_0']
        adjs = load_data['arr_1']
        chem_props = load_data['arr_2']
        data_list = []

        for x, adj, chem_prop in zip(xs, adjs, chem_props):
            x = atom_number_to_one_hot(x, self.dataset)
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=chem_prop[1:], smiles=chem_prop[0])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class FromNetworkx(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        if self.dataset == 'ego':
            return ['ego_small.pkl']
        elif self.dataset == 'community':
            return ['community_small.pkl']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']
        else:
            raise NotImplementedError()

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'ego':
            download_url('https://drive.switch.ch/index.php/s/KezKAJHY4bWNl9E/download', self.raw_dir,
                         filename='ego_small.pkl')
        elif self.dataset == 'community':
            download_url('https://drive.switch.ch/index.php/s/SLDFLYSBgsfV0ZA/download', self.raw_dir,
                         filename='community_small.pkl')

        elif self.dataset == 'enzymes':
            download_url('https://drive.switch.ch/index.php/s/dGo2OUFmOIqqDNt/download', self.raw_dir,
                         filename='ENZYMES.pkl')

    @property
    def processed_file_names(self):
        if self.dataset == 'ego':
            return ['ego_data.pt']
        elif self.dataset == 'community':
            return ['community_data.pt']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']

    def process(self):
        filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(filepath, 'rb') as pickle_file:
            graph_list = pickle.load(pickle_file)

        data_list = []
        for g in graph_list:
            data = from_networkx(g)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


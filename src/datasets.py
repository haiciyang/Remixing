import os
import glob
import musdb
import torch
# import librosa
import argparse
import numpy as np
from torch.utils import data
from IPython.display import clear_output

MUS_PERM_ID = np.array([ 6, 48, 10, 28, 43, 44,  5, 34,  9, 35, 31, 13, 33, 20,  8, 18, 17,
        1, 38, 29, 16,  0, 24, 15, 47, 19, 36, 14, 12,  3,  7, 26, 39, 41,
       37, 21, 27,  4, 23, 42, 32, 49,  2, 22, 30, 40, 25, 45, 11, 46])
MUS_VALID_ID = MUS_PERM_ID[:25]
MUS_EVAL_ID = MUS_PERM_ID[25:]

class MUSDB_data(data.Dataset):
    
    def __init__(self, dataset, n_src, with_silent):
        
        self.dataset = dataset
        if with_silent:   
            self.path = '../Data/MUSDB/large_'
        else:
            self.path = '../Data/MUSDB/'
        config = torch.load(self.path + dataset + '/config.pt')
        self.global_max = 23.3 # max of test and train
        self.all_length = config['all_length']
#         if dataset == 'test':
#             self.all_length = np.array(self.all_length)[MUS_VALID_ID]
        self.n_src = n_src
        
#         self.index_list = {}
#         base = 0
#         for file_id in range(len(self.all_length)):
#             for local_id in range(self.all_length[file_id]):
#                 self.index_list[local_id+base] = [file_id, local_id]
#             base += int(self.all_length[file_id])
            
    def __len__(self):
        
        if self.dataset == 'test':
            return len(self.all_length)//2
        else:
            return len(self.all_length)

    def __getitem__(self, idx):

        if self.dataset == 'test':
            idx = MUS_VALID_ID[idx]
        file_path = self.path + self.dataset + '/mus_' + self.dataset + '_' + str(idx) + '.pt'
        sources = torch.load(file_path) # (n, srcs, L)
        N = len(sources)
        rand_id = int(torch.rand(1)*N)
        picked = sources[rand_id, :self.n_src, :] # (n_src, L)
        
        return picked/self.global_max
    

class Slakh_data(data.Dataset):
    
    def __init__(self, dataset, n_src):
        
        self.dataset = dataset
        self.path = '../Data/Slakh/' + dataset + '/'
        if n_src == 5:
            self.path = '../Data/Slakh/' + dataset + '_5/'
        config = torch.load(self.path + 'config.pt')
        self.global_max = 53.1 # max of test and train
        self.all_length = config['all_length']
        self.total_file = config['total_file']
        self.n_src = n_src
        self.data_length = len(self.all_length)
        if dataset == 'test':
            self.data_length = self.data_length//2
        self.file_idx = [i for i in range(self.data_length) if self.all_length[i] != 0]

        
    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        
        f_idx = self.file_idx[idx]
        track_path = self.path + str(f_idx) + '.pt'
        sources = torch.load(track_path) # (n, 4, L)
        
        N = len(sources)
        rand_id = int(torch.rand(1)*N)
#         print(sources.shape)
        picked = sources[rand_id, :self.n_src, :] # (n_src, L)
        
        return picked/self.global_max
    

# class MUSDB_eval(data.Dataset):
    
#     def __init__(self, dataset, n_src):
        
        
        
        
# class Slakh_eval(data.Dataset):
    
#     def __init__(self, dataset, n_src):
        

                
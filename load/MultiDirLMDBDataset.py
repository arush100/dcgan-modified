from torch.utils.data.dataset import Dataset
import os
import numpy as np 
from load.LMDBDataset import LMDBDataset

class MultiDirLMDBDataset(Dataset):
    def __init__(self,root_dir,transform=None,zfill_len=8,name_suffix='_INTEGRATED_',req_substr='train'):
        self.root_dir = root_dir
        self.dbs = [LMDBDataset(os.path.join(self.root_dir,x),transform,zfill_len,name_suffix) for x in os.listdir(self.root_dir) if req_substr in x]
        
        self.lens = [len(x) for x in self.dbs]
        self._len = np.sum(self.lens)
    
    def __getitem__(self,ind):
        sum_of_blocks=0
        for i,l in enumerate(self.lens):
            if ind-sum_of_blocks < l: 
                break 
            sum_of_blocks += l
        
        return self.dbs[i][ind-sum_of_blocks],i
        
    def __len__(self):
        return self._len




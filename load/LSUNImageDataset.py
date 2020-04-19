from torch.utils.data import Dataset 
import os
from PIL import Image


class LSUNImageDataset(Dataset):
    def __init__(self,img_dir,dir_size=1000,transform=None):
        self.img_dir = img_dir
        self.dir_size = dir_size
        self.transform = transform
        
        subdirs = os.listdir(self.img_dir)
        self.n_subdirs = len(subdirs)
        self._len = (self.n_subdirs-1) * (self.dir_size) + len(os.listdir(os.path.join(self.img_dir,sorted(subdirs)[-1]))) 
        
    def _index_to_path(self,i):
        return os.path.join(self.img_dir,str(i//self.dir_size).zfill(4),('%i.webp'%i).zfill(12))

    def __getitem__(self, i):
        #print(i)
        return Image.open(self._index_to_path(i)) if self.transform is None else self.transform(Image.open(self._index_to_path(i)))

    def __len__(self):
        return self._len


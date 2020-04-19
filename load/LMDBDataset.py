from torch.utils.data.dataset import Dataset
import lmdb
from PIL import Image
import io 


class LMDBDataset(Dataset):
    def __init__(self,db_dir,transform=None,zfill_len=8,name_suffix='_INTEGRATED_'):
        self.db_dir = db_dir
        self.transform = transform
        self.zfill_len = zfill_len
        self.name_suffix = name_suffix
        
        self.env = lmdb.Environment(self.db_dir,readonly=True) 
        self._len = self.env.stat()['entries']
        
    def key_name(self,i):
        return (str(i).zfill(self.zfill_len) + self.name_suffix).encode()
    
    def __getitem__(self,i):
        with self.env.begin(write=False) as txn:
            img_bytes = io.BytesIO(txn.get(self.key_name(i)))
            return Image.open(img_bytes) if self.transform is None else self.transform(Image.open(img_bytes))

    def __len__(self):
        return self._len

    
    
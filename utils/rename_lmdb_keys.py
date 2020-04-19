import lmdb
import os


def reindex(pth):
    print(pth)
    env = lmdb.open(pth,map_size=10**12)
    
    ZFILL_LEN = 8
    NAME_SUFFIX = b'_INTEGRATED_'
    
    def new_key_name(index):
        return str(index).zfill(ZFILL_LEN).encode() + NAME_SUFFIX
    
    
    rename_index = 0
    with env.begin(write=True) as txn:
        for k,v in txn.cursor():
            if k.endswith(NAME_SUFFIX) and (len(NAME_SUFFIX) + ZFILL_LEN == len(k)):
                continue
            
            new_key = new_key_name(rename_index)
            txn.delete(k)
            txn.put(new_key,v,overwrite=False)
            rename_index += 1
            print(rename_index,k,'->',new_key)
            
data_root = '/home/andrew/data/lsun/data'       
for p in os.listdir(data_root):
    if 'train' in p:
        #if 'bedroom' in p: 
        reindex(os.path.join(data_root,p))        

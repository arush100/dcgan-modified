import os 
from utils.ensure_dir import ensure_dir


rt = '/home/andrew/data/cyberextrudefaces'
dir_all = os.path.join(rt,'all')
ensure_dir(dir_all)



for d in os.listdir(rt):
    if d != 'all':
        current_dir = os.path.join(rt,d)
        for f in os.listdir(current_dir):
            new_fname = '%d-%f'.replace('%d',d).replace('%f',f)
            os.symlink(os.path.join(current_dir,f),os.path.join(dir_all,new_fname))

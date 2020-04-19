import argparse
import lmdb
from utils.ensure_dir import ensure_dir
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir','-s',type=str)
parser.add_argument('--output_dir','-o',type=str)
#parser.add_argument('--write_freq','-w',type=int,default=10000)
parser.add_argument('--create_out_dir','-c',type=bool,default=True)
parser.add_argument('--replace','-r',type=bool,default=True)
args = parser.parse_args()

source_dir = args.source_dir
output_dir = args.output_dir
#write_freq = args.write_freq
create_out_dir = args.create_out_dir
replace = args.replace

if create_out_dir:
    ensure_dir(output_dir)
else:
    assert os.path.exists(output_dir)

archive_path = os.path.join(output_dir,'images.iia')
index_path = os.path.join(output_dir,'indices.iii')
if replace:
    try:
        os.remove(archive_path)
        print('Removed existing file %s'%archive_path)
    except os.error:
        pass
    try:
        os.remove(index_path)
        print('Removed existing file %s'%index_path)
    except os.error:
        pass

archive = open(archive_path,'wb')
offset_list = []


env = lmdb.open(source_dir)
with env.begin(write=False) as txn:
    for i,(k,v) in enumerate(txn.cursor()):
        archive.write(v)
        offset_list += len(v)
        





















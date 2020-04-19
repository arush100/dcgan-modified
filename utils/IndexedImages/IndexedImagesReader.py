import os


class IndexedImagesReader:
    def __init__(self,iii_path=None,iia_path=None,directory=None):
        if (iii_path is None) and (iia_path is None):
            assert (directory is not None)
            dir_contents = os.listdir(directory)
            iii_candidates = [x for x in dir_contents if x.endswith('.iii')]
            assert len(iii_candidates) == 1
            iia_candidates = [x for x in dir_contents if x.endswith('.iia')]
            assert len(iia_candidates) == 1
            
            self.iii_path = os.path.join(directory,iii_candidates[0])
            self.iia_path = os.path.join(directory,iia_candidates[0])
        
    def _read_iia(self):
        
            
    def __getitem__(self,i):
        
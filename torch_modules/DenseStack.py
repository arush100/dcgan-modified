import torch


class DenseStack(torch.nn.Module):
    def __init__(self,shp,nonlin):
        super().__init__()
        
        self.shp = shp
        self.depth = len(self.shp) - 1
        if not isinstance(nonlin,torch.nn.Module):
            assert isinstance(nonlin,list) or isinstance(nonlin,tuple)
            assert len(nonlin) == self.depth
            nonlin_list = list(nonlin)
        else:
            nonlin_list = [nonlin for _ in range(self.depth)]
        
        self.lin_layers = torch.nn.ModuleList([torch.nn.Linear(self.shp[i],self.shp[i+1]) for i in range(self.depth)])
        self.nonlin_layers = torch.nn.ModuleList(nonlin_list)

    def forward(self,x,return_intermediates=False):
        if return_intermediates:
            out = [x]
        for n,l in zip(self.nonlin_layers,self.lin_layers):
            x = l(x)
            x = n(x)
            if return_intermediates:
                out.append(x)
        return out if return_intermediates else x
    

    
    
    
    
    
    
    
    
        


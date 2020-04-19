import torch


class ConvStack(torch.nn.Module):
    def __init__(self,channel_sizes,kernel_sizes,nonlin,stride=None,padding=None,batchnorm=True,dropout=None):
        super().__init__()

        self.n_layers = len(channel_sizes) - 1
        assert len(kernel_sizes) == self.n_layers

        self.channel_sizes = channel_sizes
        self.kernel_sizes = kernel_sizes
        if not isinstance(nonlin,torch.nn.Module):
            assert isinstance(nonlin,list) or isinstance(nonlin,tuple)
            assert len(nonlin) == self.n_layers
            nonlin_list = list(nonlin)
        else:
            nonlin_list = [nonlin for _ in range(self.n_layers)]
        if stride is None:
            self.stride = [1 for _ in range(self.n_layers)]
        else:
            assert len(stride) == self.n_layers
            self.stride=stride
        if padding is None:
            self.padding = [0 for _ in range(self.n_layers)]
        else:
            assert len(padding) == self.n_layers
            self.padding=padding
        self.batchnorm = batchnorm
        self.dropout = dropout
        if self.dropout is not None:
            if not (isinstance(self.dropout,list) or isinstance(self.dropout,tuple)):
                self.dropout = [self.dropout for _ in range(self.n_layers)] 
            else:
                assert len(self.dropout) == self.n_layers
        
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(channel_sizes[i],channel_sizes[i+1],kernel_sizes[i],stride=self.stride[i],padding=self.padding[i]) for i in range(self.n_layers)])
        self.nonlin_layers = torch.nn.ModuleList(nonlin_list)
        if self.batchnorm:
            self.batchnorm_layers = torch.nn.ModuleList([torch.nn.BatchNorm2d(channel_sizes[i+1],affine=False) for i in range(self.n_layers)])
        if self.dropout:
            self.dropout_layers = torch.nn.ModuleList([torch.nn.Dropout2d(p) for p in self.dropout])


    def forward(self,imgs):
        for i in range(self.n_layers):
            imgs = self.nonlin_layers[i](self.conv_layers[i](imgs))
            if self.batchnorm:
                imgs = self.batchnorm_layers[i](imgs)
            if self.dropout:
                imgs = self.dropout_layers[i](imgs)

        return imgs



    
    
    
    
    
    
    
        


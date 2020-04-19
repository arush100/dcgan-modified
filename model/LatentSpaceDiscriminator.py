import torch
from torch_modules.View import View

class LatentSpaceDiscriminatorLayer(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,batchnorm=True,activation=None,dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.convt_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size,self.stride,self.padding,bias=False)
        if batchnorm:
            self.batchnorm_layer = torch.nn.BatchNorm2d(out_channels,affine=True)
        if activation is not None:
            self.activation_layer = activation
        if dropout:
            self.dropout = dropout
            self.dropout_layer = torch.nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.convt_layer(x)
        if hasattr(self,'batchnorm_layer'):
            x = self.batchnorm_layer(x)
        if hasattr(self,'activation_layer'):
            x = self.activation_layer(x)
        if hasattr(self,'dropout'):
            x = self.dropout_layer(x)
        return x
    
class LatentSpaceDiscriminator(torch.nn.Module):
    def __init__(self,channels,kernel_sizes,strides,padding,dropout):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.dropout = dropout 
        
        self.depth = len(self.kernel_sizes)
        assert len(self.channels) == self.depth + 1
        assert len(self.strides) == self.depth 
        assert len(self.padding) == self.depth
        if isinstance(self.dropout,float) or isinstance(self.dropout,int):
            self.dropout = [self.dropout] * (self.depth-1)
        assert len(self.dropout) == self.depth - 1
        self.latent_space_dims = self.channels[0] 
    

        self.inner_layers = torch.nn.ModuleList([LatentSpaceDiscriminatorLayer(self.channels[i],
                                                                           self.channels[i+1],
                                                                           self.kernel_sizes[i],
                                                                           self.strides[i],
                                                                           self.padding[i],
                                                                           True,
                                                                           torch.nn.LeakyReLU(0.2,True),
                                                                           self.dropout[i])
                                                for i in range(0,self.depth-1)])
        self.output_layer = LatentSpaceDiscriminatorLayer(self.channels[-2],
                                                      self.channels[-1],
                                                      self.kernel_sizes[-1],
                                                      self.strides[-1],
                                                      self.padding[-1],
                                                      False,
                                                      torch.nn.Sequential(View([-1,channels[-1]]),torch.nn.LogSoftmax(dim=-1)))
    
    def forward(self,x):
        for layer in self.inner_layers:
            x = layer(x)
        return self.output_layer(x)

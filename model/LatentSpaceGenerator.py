import torch

class LatentSpaceGeneratorLayer(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,batchnorm=True,
                 activation=None,dropout=0,activation_histogram_buffer_name=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.convt_layer = torch.nn.ConvTranspose2d(in_channels,out_channels,kernel_size,self.stride,self.padding,bias=False)
        if batchnorm:
            self.batchnorm_layer = torch.nn.BatchNorm2d(out_channels,affine=True)
        if activation is not None:
            self.activation_layer = activation
        if dropout:
            self.dropout = dropout
            self.dropout_layer = torch.nn.Dropout2d(dropout)
        if activation_histogram_buffer_name is not None:
            self.activation_histogram_buffer_name = activation_histogram_buffer_name
            self.register_buffer(activation_histogram_buffer_name,None)
            #other is (min,max,numel,sum,sum of squares)
            self.register_buffer(activation_histogram_buffer_name+'_other',None)
        
    def forward(self, x):
        x = self.convt_layer(x)
        if hasattr(self,'batchnorm_layer'):
            x = self.batchnorm_layer(x)
        if hasattr(self,'activation_layer'):
            x = self.activation_layer(x)
        if hasattr(self,'activation_histogram_buffer_name'):
            setattr(self,self.activation_histogram_buffer_name,torch.histc(x))
            setattr(self,self.activation_histogram_buffer_name+'_other',
                    torch.Tensor([x.min().item(),x.max().item(),x.numel(),x.sum().item(),x.view(-1).dot(x.view(-1)).item()]))
        if hasattr(self,'dropout'):
            x = self.dropout_layer(x)
        return x
    
class LatentSpaceGenerator(torch.nn.Module):
    def __init__(self,channels,kernel_sizes,strides,padding,dropout,draw_activation_hist):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.dropout = dropout 
        self.draw_activation_hist = draw_activation_hist
        
        self.depth = len(self.kernel_sizes)
        assert len(self.channels) == self.depth + 1
        assert len(self.strides) == self.depth 
        assert len(self.padding) == self.depth
        if isinstance(self.dropout,float) or isinstance(self.dropout,int):
            self.dropout = [self.dropout] * (self.depth-1)
        assert len(self.dropout) == self.depth - 1
        self.latent_space_dims = self.channels[0] 
        
        self.initial_projection_layer = LatentSpaceGeneratorLayer(self.channels[0],
                                                                  self.channels[1],
                                                                  self.kernel_sizes[0],
                                                                  self.strides[0],
                                                                  self.padding[0],
                                                                  False,
                                                                  None,
                                                                  self.dropout[0],
                                                                  'activation_histogram_0' if self.draw_activation_hist else None
                                                                  )
        
        self.inner_layers = torch.nn.ModuleList([LatentSpaceGeneratorLayer(self.channels[i],
                                                                           self.channels[i+1],
                                                                           self.kernel_sizes[i],
                                                                           self.strides[i],
                                                                           self.padding[i],
                                                                           True,
                                                                           torch.nn.LeakyReLU(0.2,True),
                                                                           self.dropout[i],
                                                                           'activation_histogram_%d'%i if self.draw_activation_hist else None)
                                                for i in range(1,self.depth-1)])
        

        
        self.output_layer = LatentSpaceGeneratorLayer(self.channels[-2],
                                                      self.channels[-1],
                                                      self.kernel_sizes[-1],
                                                      self.strides[-1],
                                                      self.padding[-1],
                                                      False,
                                                      torch.nn.Sigmoid(),
                                                      0.,
                                                      'activation_histogram_%d'%(self.depth-1) if self.draw_activation_hist else None)
    
    def forward(self,z):
        x = self.initial_projection_layer(z)
        for layer in self.inner_layers:
            x = layer(x)
        return self.output_layer(x)

        
        
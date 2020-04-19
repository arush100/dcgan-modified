import torch

class Reshape(torch.nn.Module):
    def __init__(self,new_shape):
        torch.nn.Module.__init__(self)
        self.new_shape = new_shape
        
    def forward(self, x):
        return x.reshape(*self.new_shape)
    
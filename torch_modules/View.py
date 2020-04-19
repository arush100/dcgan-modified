import torch

class View(torch.nn.Module):
    def __init__(self,new_shape):
        super().__init__()
        self.new_shape = new_shape
    
    def forward(self, x):
        return x.view(*self.new_shape)
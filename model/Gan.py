import torch

class Gan(torch.nn.Module):
    def __init__(self,generator,discriminator):
        super().__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        
    def forward(self,*args,**kwargs):
        try:
            only_g = kwargs.pop('only_g')
        except KeyError:
            only_g = False
        try:
            only_d = kwargs.pop('only_d')
        except KeyError:
            only_d = False
            
        assert not(only_g and only_d)
        
        if only_g: 
            return self.generator(*args,**kwargs)
        if only_d:
            return self.discriminator(*args,**kwargs)
        
        generator_output = self.generator(*args,**kwargs)
        return generator_output,self.discriminator(generator_output)
        
        
from torch.nn.parallel.distributed import DistributedDataParallel
class DistributedDataParallelGAN(DistributedDataParallel):
    '''
    Because typing gan.module.generator or gan.module.discriminator is a pain and nobody deserves to suffer that much. 
    '''
    @property
    def generator(self):
        return self.module.generator
    @property
    def discriminator(self):
        return self.module.discriminator
from utils.TrainingHooks.TrainingHook import TrainingHook
import warnings
from utils.ensure_dir import ensure_dir
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.tensorboard.writer import SummaryWriter
warnings.filterwarnings('default',category=FutureWarning)


class DiscriminatorTensorboardHook(TrainingHook):
    def __init__(self,log_dir):
        super().__init__()
        self.log_dir = ensure_dir(log_dir)

        self.writer = SummaryWriter(self.log_dir)
        #self.writer.add_custom_scalars(_LAYOUT)


    def call(self,*args,**kwargs):
        self.writer.add_scalar('Loss/discriminator_fake',kwargs['discriminator_loss_fake_cpu'],self.step)
        self.writer.add_scalar('Loss/discriminator_real',kwargs['discriminator_loss_real_cpu'],self.step)

        self.notify('Writing real discriminator loss %.4f, fake discriminator loss %.4f.'%(kwargs['discriminator_loss_real_cpu'].item(),
                                                                                           kwargs['discriminator_loss_fake_cpu'].item()))
        

        '''
        for p in kwargs['model'].discriminator.parameters(): 
            self.notify('Parameter of size %d has mean grad abs %.5f'%(p.numel(),p.grad.abs().mean()))
        '''
            
            
            
            
            
            
            

from utils.TrainingHooks.TrainingHook import TrainingHook
import warnings
from utils.ensure_dir import ensure_dir
from utils.write_conv_kernel_image_summaries import write_conv_kernel_to_image_summaries
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.tensorboard.writer import SummaryWriter
warnings.filterwarnings('default',category=FutureWarning)
import torch



class GeneratorTensorboardHook(TrainingHook):
    def __init__(self,log_dir,img_log_freq=10,max_batchsize=16):
        super().__init__()
        self.log_dir = ensure_dir(log_dir)
        self.img_log_freq = img_log_freq
        self.max_batchsize = max_batchsize

        self.writer = SummaryWriter(self.log_dir)
        #self.writer.add_custom_scalars(_LAYOUT)


    def _image_transform(self,img):
        nneg_img = torch.abs(img)
        return 255 * nneg_img/nneg_img.sum()
        

    def call(self,*args,**kwargs):
        self.writer.add_scalar('Loss/generator',kwargs['generator_loss_cpu'],self.step)
        self.notify('Writing loss %s.'%round(kwargs['generator_loss_cpu'].item(),5))

        self.writer.add_scalar('LossVariance/generator',kwargs['generator_batch_loss_var_cpu'],self.step)
        if kwargs['draw_activation_hist']:
            self._write_activation_histograms(buffer_dict=kwargs['buffers_cpu'])
        
        
        '''
        for p in kwargs['model'].generator.parameters(): 
            self.notify('Parameter of size %d has mean grad abs %.5f'%(p.numel(),p.grad.abs().mean()))
        '''
        '''
        (as of up-to-date torch and numpy on 9/7/19)
        In the function torch.utils.tensorboard._utils.make_grid, a grid is initialized with np.zeros, with no dtype given.
        It defaults to float, and in the next line from the function where this function is called, in torch.utils.tensorboard.summary.image,
        a scale factor is determined based on the dtype of the returned value from make_grid. 
        So basically if you try to pass uint8 it gets treated as float anyway, because of the
        default behavior of np.zeros. The scale factor then is 255 instead of 1. The easiest way to fix this 
        is just divide by 255 here if the desired behavior (as determined by the dtype of images passed) was to 
        treat them as uint8. Ends up creating a conversion from float32 to uint8 (out of the model) right back to float32, but 
        I'd rather just stuff this in here to maintain my sanity elsewhere and be able to pretend I'm using 
        uint8 when I want to use it. 
        '''
        
        
        
        if (self.step)%self.img_log_freq==0:
            self.notify('Sending batch images to tensorboard...')
            if kwargs['generated_images_cpu'].dtype == torch.uint8:
                self.writer.add_images('Generated Images',kwargs['generated_images_cpu'][:self.max_batchsize].to(torch.float32)/255.,self.step)
                self.writer.add_images('Sampled Images',kwargs['sampled_images_cpu'][:self.max_batchsize].to(torch.float32)/255.,self.step)
            else:
                self.writer.add_images('Generated Images',kwargs['generated_images_cpu'][:self.max_batchsize],self.step)
                self.writer.add_images('Sampled Images',kwargs['sampled_images_cpu'][:self.max_batchsize],self.step)
            self.notify('Done. Sent %d images.'%(min(kwargs['generated_images_cpu'].shape[0],self.max_batchsize)
                                                 +min(kwargs['sampled_images_cpu'].shape[0],self.max_batchsize)))
        
            self.notify('Sending last layer histogram to tensorboard...')
            self.writer.add_histogram('Generator/Last Layer',kwargs['model'].generator.output_layer.convt_layer.weight,self.step)
            self.notify('Done.')
            
            self._write_batchnorm_histograms(kwargs['model'].generator)

            
            '''
            self.notify('Sending layer 0 filter weights to tensorboard...')
            conv_0_weight = kwargs['model'].generator.conv.conv_layers[0].weight
            write_conv_kernel_to_image_summaries(self.writer,'Generator/First_Kernel',conv_0_weight,self.step,self._image_transform)
            self.notify('Done.')
            
            self.notify('Sending layer -1 filter weights to tensorboard...')
            conv_final_weight = kwargs['model'].generator.conv.conv_layers[-1].weight
            write_conv_kernel_to_image_summaries(self.writer,'Generator/Last_Kernel',conv_final_weight,self.step,self._image_transform)
            self.notify('Done.')
            '''
            
            '''
            max_kernel_hist = 5
            num_kernels_written = 0
            for i in range(conv_0_weight.shape[0]):
                for j in range(0,i+1):
                    self.writer.add_image('Generator/First Kernel[%i][%j]',conv_0_weight[i,j].unsqueeze(0),self.step)
                    num_kernels_written += 1
                    if num_kernels_written > max_kernel_hist:
                        break
                if num_kernels_written > max_kernel_hist:
                    break
            '''

    def _write_batchnorm_histograms(self,model):
        weights,biases,names = [],[],[]
        for i,m in enumerate(model.modules()):
            if 'BatchNorm' in m.__class__.__name__:
                if m.weight is not None:
                    weights.append(m.weight)
                    biases.append(m.bias)
                    names.append('%s_size%d_index%d'%(m.__class__.__name__,m.weight.numel(),i))
        
        self.notify('Sending %d batchnorm histograms to tensorboard...'%(len(weights)+len(biases)))
        for w,b,n in zip(weights,biases,names):
            self.writer.add_histogram('BatchNorm/%s_weight'%n,w,self.step)
            self.writer.add_histogram('BatchNorm/%s_bias'%n,b,self.step)
        self.notify('Done.')
                    
    def _write_activation_histograms(self,buffer_dict,md_suffix='_other'):
        self._write_prebinned_histograms(buffer_dict, 'activation_histogram',md_suffix)
                    
    def _write_prebinned_histograms(self,buffer_dict,identifying_substr,md_suffix='_other'):
        num_histograms_written=0
        self.notify('Writing activation histograms...')
        for k,v in buffer_dict.items():
            if (identifying_substr in k) and not k.endswith(md_suffix):
                #md_suffix should be [min,max,numel,sum,sum of squares]
                hist_min,hist_max,hist_num,hist_sum,hist_sum_squares = buffer_dict[k+md_suffix]
                step_size = float(hist_max-hist_min)/v.numel()
                bucket_limits = torch.cat((hist_min+torch.arange(v.numel()-1)*step_size,torch.Tensor([hist_max])),dim=0)
                self.writer.add_histogram_raw('Activations/%s'%k, hist_min, hist_max, hist_num, 
                                               hist_sum, hist_sum_squares,bucket_limits,v,self.step)
                num_histograms_written += 1
                
        self.notify('Done. Wrote %d histograms'%num_histograms_written)
        
                    



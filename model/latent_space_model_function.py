import torch
from math import ceil
from model.Gan import Gan
from model.LatentSpaceDiscriminator import LatentSpaceDiscriminator
from model.LatentSpaceGenerator import LatentSpaceGenerator

def latent_space_model_function(**kwargs):
    cuda = kwargs['cuda']
    image_shape = kwargs['image_shape']
    n_lsun_classes = kwargs['n_lsun_classes']
    draw_activation_hist = kwargs['draw_activation_hist']
    
    ############ generator
    if cuda:
        
        generator_channels = [100] + [1024,512,256,128] + [3]
        generator_kernel_sizes = [6,8,10,12,14] 
        generator_strides = [1] + [2]*4 
        generator_padding = [0] + [1]*4
        generator_dropout = 0.
        
        
    else:
        generator_channels = [100] + [51,25,12,10] + [3]
        generator_kernel_sizes = [6,8,10,12,14] 
        generator_strides = [1] + [2]*4 
        generator_padding = [0] + [1]*4
        generator_dropout = 0.


    generator = LatentSpaceGenerator(channels=generator_channels,
                                     kernel_sizes=generator_kernel_sizes,
                                     strides=generator_strides,
                                     padding=generator_padding,
                                     dropout=generator_dropout,
                                     draw_activation_hist=draw_activation_hist)
    
    ############ disc
    if cuda:
        discriminator_conv_channels = [3] + [128,256,512,1024,n_lsun_classes+1]
        discriminator_conv_kernel_sizes = [14,12,10,8,6] 
        discriminator_conv_stride=[2]*4 + [1]
        discriminator_conv_padding=[1]*4 + [0]
        
                    
        _out_shape = image_shape[1:] 
        for kernel,stride in zip(discriminator_conv_kernel_sizes,discriminator_conv_stride):
            _out_shape = [ceil((x-kernel+1)/stride) for x in _out_shape]
        
        discriminator_conv_dropout = 0.5

    
    
    else:
        discriminator_conv_channels = [3] + [12,25,51,102,1]

        discriminator_conv_kernel_sizes = [4]*5
        discriminator_conv_stride=[2]*4 + [1]
        discriminator_conv_padding=[1]*4 + [0]
                
        _out_shape = image_shape[1:] 
        for kernel,stride in zip(discriminator_conv_kernel_sizes,discriminator_conv_stride):
            _out_shape = [ceil((x-kernel+1)/stride) for x in _out_shape]
        
        discriminator_conv_dropout = 0.
        

    discriminator = LatentSpaceDiscriminator(discriminator_conv_channels,discriminator_conv_kernel_sizes,
                                             discriminator_conv_stride,discriminator_conv_padding,discriminator_conv_dropout)
    
    ###########


    return Gan(generator,discriminator)




import argparse
import torch
from PIL import Image
import os
from model.LatentSpaceGenerator import LatentSpaceGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',type=str,help='Location of the saved model to use.')
    parser.add_argument('output_dir',type=str,help='Directory to put the generated images in.')
    parser.add_argument('-n','--num_images',type=int,default=5,help='Number of images to generate. Default: 5.')
    parser.add_argument('-c','--cuda',action='store_true',help='Enables cuda.')
    parser.add_argument('-l','--latent_dims',type=int,default=100,help='Dimensionality of the latent space. \
    Default: 100 (the value required for the demo model).')
    parser.add_argument('-p','--prefix',type=str,default='demo-image',help='Prefix for the filenames of the generated images.\
    Default: "demo-image".')
    parser.add_argument('-w','--walk',type=float,default=0.,help='Step size of the walk through latent space. \
    If 0, images will generated by uniformly sampling from the latent space, instead of performing a walk. Default: 0 (no walk).')
    parser.add_argument('-s','--seed',type=int,default=-1,help='Random seed for reproducibility. If a value less than 0 is given, \
    no seeding is done. Default: -1 (no seeding).')
    
    args = parser.parse_args()
   
    assert args.walk>=0,'Pass a nonnegative step size.'
   
    if args.seed >= 0:
        torch.manual_seed(args.seed)
   
    device = torch.device('cuda') if args.cuda else torch.device('cpu') 


    generator_channels = [100] + [1024,512,256,128] + [3]
    generator_kernel_sizes = [6,8,10,12,14]
    generator_strides = [1] + [2]*4 
    generator_padding = [0] + [1]*4
    generator_dropout = 0.


    generator = LatentSpaceGenerator(channels=generator_channels,
                                     kernel_sizes=generator_kernel_sizes,
                                     strides=generator_strides,
                                     padding=generator_padding,
                                     dropout=generator_dropout,
                                     draw_activation_hist=False)    
    generator.load_state_dict(torch.load(args.model_path,map_location=device))
    generator.eval()

    if args.walk > 0:
        noise = torch.rand(2,args.latent_dims)
        latent_input = torch.stack([noise[0] for _ in range(args.num_images)],dim=0) + torch.stack([i * args.walk * noise[1] for i in range(args.num_images)])
        latent_input = latent_input.clamp(0,1).reshape(args.num_images,args.latent_dims,1,1)
    else:
        latent_input = torch.rand(args.num_images,args.latent_dims,1,1)

    print('Generating %d images...'%args.num_images)
    if not args.cuda:
        print('(since you\'re doing this on a cpu, this might require a little patience on your part)')
    with torch.no_grad():
        images = 255 * generator(latent_input.to(device))
    images = images.to('cpu').to(torch.uint8).numpy().transpose(0,2,3,1)
    print('Done.')
    
    print('Writing images to %s...'%args.output_dir)
    for i in range(args.num_images):
        img = Image.fromarray(images[i])
        img.save(os.path.join(args.output_dir,args.prefix+'{}.jpg'.format(str(i).zfill(5))))
    print('Done.')
    
    
    
    
    
    
    
    
    
    
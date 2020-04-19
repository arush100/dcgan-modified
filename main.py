import sys 
import torch
import numpy as np 
import argparse
from utils.ensure_dir import ensure_dir
from datetime import datetime
import os
import torchvision
from torch.utils.data.dataloader import DataLoader
from utils.TrainingHooks.TrainingHookList import TrainingHookList
from hooks.ModelSaveHook import ModelSaveHook
from hooks.GeneratorTensorboardHook import GeneratorTensorboardHook
from hooks.DiscriminatorTensorboardHook import DiscriminatorTensorboardHook
from torch_modules.DistributedDataParallelGAN import DistributedDataParallelGAN
from model.latent_space_model_function import latent_space_model_function
from utils.param_count import param_count
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from load.LMDBDataset import LMDBDataset
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.conv import Conv2d
from torchvision.datasets.lsun import LSUN
from load.MultiDirLMDBDataset import MultiDirLMDBDataset

DISCRIMINATOR_NAME = 'Discriminator'
GENERATOR_NAME = 'Generator'

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',type=str)
parser.add_argument('--output_root_dir',type=str)
parser.add_argument('--local_rank', type=int, default=0)  
parser.add_argument('--verbose',dest='verbose',action='store_true')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--latent_space_dims',type=int,default=100)
parser.add_argument('--save_interval',type=int,default=2000)
parser.add_argument('--load_path_generator',type=str,default='')
parser.add_argument('--load_path_discriminator',type=str,default='')

parser.set_defaults(verbose=True)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
cuda = n_gpu > 0
if args.local_rank == 0:
    print('Running on %d GPUs.'%n_gpu if n_gpu!=1 else 'Running on 1 GPU.')


############ hyperparameters, settings, etc

image_shape = [3,178,178]
total_image_size = np.prod(image_shape)
lr = [2e-4,2e-4]
betas = [(0.5,0.999),(0.5,0.999)]
device = torch.device('cuda:%d'%args.local_rank) if cuda else torch.device('cpu')
draw_activation_hist = True
############

############ process setup, for distributed 
if cuda: 
    torch.cuda.set_device(args.local_rank)
if n_gpu > 1:
    torch.distributed.init_process_group('nccl',init_method='env://',world_size=n_gpu,rank=args.local_rank)

if args.local_rank==0:
    output_root_dir = args.output_root_dir
    ensure_dir(output_root_dir)
    
    t = datetime.now()
    run_id = '{}-{}-{}@{}:{}:{}@proc:{}@device:{}'.format(t.year,str(t.month).zfill(2),str(t.day).zfill(2),
                                                          str(t.hour).zfill(2),str(t.minute).zfill(2),str(t.second).zfill(2),
                                                          args.local_rank,device)
    run_dir = os.path.join(output_root_dir,run_id)
    ensure_dir(run_dir)
############


############ data
transforms = [torchvision.transforms.Resize(image_shape[-2:]),
              torchvision.transforms.ToTensor(),
              ]
transform = torchvision.transforms.Compose(transforms)

def create_ds():
    return MultiDirLMDBDataset(args.image_dir,transform=transform)

image_dataset = create_ds()
n_lsun_classes = len(image_dataset.dbs)
if args.local_rank == 0:
    print('Found %d classes of real images in subdirs %s.'%(n_lsun_classes,[x.db_dir for x in image_dataset.dbs]))

if cuda:
    image_sampler = DistributedSampler(image_dataset,n_gpu,args.local_rank,shuffle=True)
else:
    image_sampler = RandomSampler(image_dataset)


def collate_fn(data):
    x,y = zip(*data)
    return torch.stack(x,dim=0),torch.tensor(y,dtype=torch.long)


loader = DataLoader(image_dataset,
                    batch_size=args.batchsize,
                    sampler=image_sampler,
                    pin_memory=cuda,
                    collate_fn=collate_fn,
                    num_workers=8,
                    drop_last=True)


############



############ model
model = latent_space_model_function(cuda=cuda,image_shape=image_shape,n_lsun_classes=n_lsun_classes,draw_activation_hist=draw_activation_hist)

def initializer(m):
    classname = m.__class__.__name__
    if isinstance(m, ConvTranspose2d) or isinstance(m, Conv2d):
        torch.nn.init.normal_(m.weight,mean=0.,std=0.02)
        if args.local_rank==0:
            print('Initializing %s layer ~ N(0,0.02).'%classname)
    elif 'BatchNorm' in classname: 
        if m.weight is not None: 
            torch.nn.init.normal_(m.weight,mean=1.,std=0.02)
            if args.local_rank==0:
                print('Initializing %s layer weight ~ N(1,0.02).'%classname)
            torch.nn.init.constant_(m.bias,0.)
            if args.local_rank==0:
                print('Initializing %s layer bias to 0.'%classname)

if n_gpu > 1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
if cuda:
    model = model.to(device)
if n_gpu > 1:
    model = DistributedDataParallelGAN(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)




if len(args.load_path_generator): 
    if args.local_rank==0:
        print('Loading generator from %s...'%(args.load_path_generator))
    model.generator.load_state_dict(torch.load(args.load_path_generator,map_location=device),strict=False)
else:
    model.generator.apply(initializer)
    
    
if len(args.load_path_discriminator): 
    if args.local_rank==0:
        print('Loading discriminator from %s...'%(args.load_path_discriminator))
    model.discriminator.load_state_dict(torch.load(args.load_path_discriminator,map_location=device))
else:
    model.discriminator.apply(initializer)
############


    


############ losses and optimizers
generator_loss_fn = torch.nn.NLLLoss(reduction='none')  
generator_optimizer = torch.optim.Adam(model.generator.parameters(),
                                       lr=lr[0],
                                       betas=betas[0]
                                       )

discriminator_loss_fn = torch.nn.NLLLoss(reduction='none')
discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(),
                                           lr=lr[1],
                                           betas=betas[1]
                                           )
############

############ training hooks 
if args.local_rank == 0:
    tb_dir = ensure_dir(os.path.join(run_dir,'tb'))
    file_path = os.path.realpath(__file__)
    project_path = os.path.split(file_path)[0]
    generator_training_hooks = TrainingHookList(GeneratorTensorboardHook(os.path.join(tb_dir,GENERATOR_NAME)),
                                                  ModelSaveHook(model.generator,GENERATOR_NAME,os.path.join(run_dir,GENERATOR_NAME),args.save_interval,-1).setup(os.path.join(project_path,'model/LatentSpaceGenerator.py'),
                                                                                                                                                            os.path.join(project_path,'model/latent_space_model_function.py'))
                                                  )
    
    discriminator_training_hooks = TrainingHookList(DiscriminatorTensorboardHook(os.path.join(tb_dir,DISCRIMINATOR_NAME)),
                                                  ModelSaveHook(model.discriminator,DISCRIMINATOR_NAME,os.path.join(run_dir,DISCRIMINATOR_NAME),args.save_interval,-1).setup(os.path.join(project_path,'model/Discriminator.py'),
                                                                                                                                                                        os.path.join(project_path,'model/latent_space_model_function.py'))
                                                  )
    
############
if args.local_rank==0:
    print('='*70)
    print(model)
    print('='*70)
    print
    print('Generator param count: %d.'%param_count(model.generator))
    print('Discriminator param count: %d.'%param_count(model.discriminator))
    print 

for epoch in range(100):
    try:
        loader.sampler.set_epoch(epoch)
    except AttributeError:
        pass
    
    if args.local_rank == 0:
        print('######### BEGINNING EPOCH %d #########'%epoch)
    
    for i,(sampled_images_cpu,sampled_targets_cpu) in enumerate(loader):
        sampled_images = sampled_images_cpu.to(device,non_blocking=True)
        sampled_targets = sampled_targets_cpu.to(device,non_blocking=True)
        
        generator_seed = torch.rand(args.batch_size,args.latent_space_dims,1,1,device=device)
        generated_images = model(generator_seed,only_g=True)


        ############ discriminator-specific
        discriminator_predictions_real = model(sampled_images,only_d=True) 

        discriminator_batch_loss_real = discriminator_loss_fn(discriminator_predictions_real,sampled_targets)
        discriminator_loss_real = discriminator_batch_loss_real.mean()        
        discriminator_optimizer.zero_grad()
        discriminator_loss_real.backward()
    
        discriminator_predictions_fake = model(generated_images.detach(),only_d=True)
        discriminator_batch_loss_fake = discriminator_loss_fn(discriminator_predictions_fake,torch.zeros_like(sampled_targets))
        discriminator_loss_fake = discriminator_batch_loss_fake.mean()
        discriminator_loss_fake.backward()
        
        discriminator_optimizer.step()
        
        if args.local_rank == 0: 
            ############ async copy some common stuff
            discriminator_loss_real_cpu = discriminator_loss_real.to('cpu',non_blocking=True)
            discriminator_loss_fake_cpu = discriminator_loss_fake.to('cpu',non_blocking=True)
            ############
            
            discriminator_training_hooks(**locals())
        ############
    

        ############ generator-specific
        discriminator_predictions_fake_for_generator = model(generated_images,only_d=True)
        generator_batch_loss = generator_loss_fn(discriminator_predictions_fake_for_generator,torch.ones(args.batch_size,device=device,dtype=torch.long))
        generator_loss = generator_batch_loss.mean()
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        
        if args.local_rank == 0: 
            ############ async copy some common stuff
            buffers_cpu = {k:v.detach().to('cpu',non_blocking=True) for k,v in model.generator.named_buffers()}

            if i%generator_training_hooks[0].img_log_freq==0:
                generated_images_cpu = generated_images[:16].to('cpu',non_blocking=True)
            
            generator_loss_cpu = generator_loss.to('cpu',non_blocking=True)
            generator_batch_loss_var_cpu = generator_batch_loss.var().to('cpu',non_blocking=True)
            ############
            generator_training_hooks(**locals())
        ############


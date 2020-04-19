import torch
def jpegify_tensor(x):
    return x.clamp(0,255).to(torch.uint8).transpose(0,1).transpose(1,2)
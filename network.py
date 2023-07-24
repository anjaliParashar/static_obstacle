#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as T
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import UNet2DModel


route_dim = 2
dim_y = 256
route_horizon = 701
dim_x = 256
# example inputs
device = torch.device('cuda')
noised_action = torch.randn((1, 1, dim_x, dim_y)).to(device)
obs = torch.zeros((1, route_horizon, route_dim)).to(device)
diffusion_iter = torch.zeros((1,)).to(device)

noise_pred_net = UNet2DModel(
    sample_size= [256,256], #config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(256, 256, 256*2, 256*2),#, 512*2, 512*2),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        #"DownBlock2D",
        #"DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        #"UpBlock2D",
        #"UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).cuda()


noise_pred = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
) #.to(device)
print(noise_pred.sample.shape, noise_pred)

"""
from diffusers import UNet2DModel

noise_pred_net = UNet2DModel(
    in_channels=1,
    out_channels=1,
    sample_size= [400,600],
    encoder_hid_dim=522*2,
).cuda()
noised_action = torch.randn((1, 400, 600)).to(device)
obs = torch.zeros((1, 522*2)).to(device)
diffusion_iter = torch.zeros((1,)).to(device)
noise_pred = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    encoder_hidden_states=obs
).to(device)
#print(noise_pred.shape, noise_pred)
"""
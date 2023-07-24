from dataset_gen import TrajectoryDataset
import torch
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as T
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataclasses import dataclass
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda')
@dataclass
class TrainingConfig:
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    seed = 0
config = TrainingConfig()

#####CREATE DATASET ################
dataset = TrajectoryDataset()
# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("size of batched dataset:", len(dataloader))
print("batch['obstacle'].shape:", batch['obstacle'].shape)
print("batch['route'].shape", batch['route'].shape)
#####################################################
#################### INITIALIZE NETWORK & SCHEDULER
####################################################
noise_pred_net = UNet2DModel(
    sample_size= [256,256], #config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(256, 256, 512, 512),#, 512*2, 512*2),  # the number of output channels for each UNet block
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

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#noise = torch.randn(batch['obstacle'].shape)
#timesteps = torch.LongTensor([50])
#noisy_image = noise_scheduler.add_noise(batch['obstacle'].shape, noise, timesteps)

optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
num_epochs = 3
#writer = SummaryWriter('runs_/diffusion_'+str(num_epochs))
#####################################################
#################### Training
####################################################

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nobstacle = nbatch['obstacle'].to(device)
                nobstacle = nobstacle.float()
                # sample noise to add to actions
                noise = torch.randn(nobstacle.shape, device=device)
                bs = nobstacle.shape[0]

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
                ).long()

                #print(timesteps.shape)
                noisy_obstacle = noise_scheduler.add_noise(
                nobstacle, noise, timesteps)
                #print(noisy_actions.shape,obs_cond.shape)
            # predict the noise residual
                noisy_obstacle.to(device)
                #print(noisy_actions.shape)
                timesteps.to(device)
                
                noise_pred = noise_pred_net(
                                sample=noisy_obstacle,
                                timestep=timesteps
                            ).sample   #.to(device)
            
            # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                #ema.step(noise_pred_net)

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        writer.add_scalar('training loss',
                np.mean(epoch_loss),
                epoch_idx)
        
        # Additional information
EPOCH = num_epochs
PATH = "static_obstacle_model_"+str(num_epochs)+".pt"
LOSS = np.mean(epoch_loss)

torch.save({
            'epoch': EPOCH,
            'model_state_dict': noise_pred_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
        


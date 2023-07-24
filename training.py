#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights
import torch
from network import ConditionalUnet1D
import numpy as np
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from dataset_gen import TrajectoryDataset
from GPUtil import showUtilization as gpu_usage
import gc
#from action_latent import Encoder, Decoder
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
gpu_usage()
gc.collect()
#actual data load
torch.cuda.empty_cache()
dataset = TrajectoryDataset()
# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)
# visualize data in batch
batch = next(iter(dataloader))
print("batch['route'].shape:", batch['route'].shape)
print("batch['obstacle'].shape", batch['obstacle'].shape)
print('dataset length:',len(dataloader))

route = batch['route'].float()
obstacle = batch['obstacle'].float()
device = torch.device('cuda')
# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights

obs_horizon = 580
pred_horizon = 512
obs_dim = 2
action_dim = 1024
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
"""
noise_pred_net = UNET(
    in_channels = 1,
    first_out_channels = 128,
    exit_channels = 1,
    downhill = 4,
    padding=0,
    global_cond_dim=obs_dim*obs_horizon
)
"""
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

noise_pred_net = noise_pred_net.cuda()
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)
# Cosine LR schedule with linear warmup
num_epochs = 100
writer = SummaryWriter('runs_/diffusion_'+str(num_epochs))

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

num_diffusion_iters = 1000
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)


with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nroute = nbatch['route'].to(device)
                nobstacle = nbatch['obstacle'].to(device)
                nroute = nroute.float()
                nobstacle = nobstacle.float()
                B = nroute.shape[0]
                
                route_cond = nroute[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                route_cond = route_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(nobstacle.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                #print(timesteps.shape)
                noisy_obstacle = noise_scheduler.add_noise(
                nobstacle, noise, timesteps)
                #print(noisy_actions.shape,obs_cond.shape)
            # predict the noise residual
                noisy_obstacle.to(device)
                #print(noisy_actions.shape)
                timesteps.to(device)
                route_cond.to(device)
                noise_pred = noise_pred_net(
                                sample=noisy_obstacle,
                                timestep=timesteps,
                                global_cond= route_cond #obs.flatten(start_dim=1)
                            ).to(device)
            #noise_pred = noise_pred_net(
            #    noisy_actions, timesteps, global_cond=obs_cond).to(device)
            #noise_pred = noise_pred
            
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
"""
                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nroute[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(nroutetacle.shape, device=device)
            
                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()
                #print('timesteps',timesteps.shape)
                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    nroutetacle, noise, timesteps)
                # predict the noise residual
                noisy_actions.to(device)
                timesteps.to(device)
                obs_cond.to(device)
                noise_pred = noise_pred_net(
                                sample=noisy_actions,
                                timestep=timesteps,
                                global_cond=obs.flatten(start_dim=1)
                            ).to(device)
                #noise_pred = noise_pred_net(
                #    noisy_actions, timesteps, global_cond=obs_cond).to(device)
                #noise_pred = noise_pred
                
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
                    sum(epoch_loss) / len(dataloader),
                    epoch_idx)
"""
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
        


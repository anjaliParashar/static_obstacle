import numpy as np
import torch
import pickle

file_0 = open('train_data/file_0.pkl', 'rb')
# dump information to that file
data0 = pickle.load(file_0)
file_0.close()  # close the file
obstacle_list = data0['obstacle']
route_list = data0['route']
print('length of collected file:', len(obstacle_list))

for i in range(4):
    file_i = "train_data/file_"+str(i+1)+".pkl" 
    file = open(file_i, 'rb')
    datai = pickle.load(file)
    file.close()
    obstacle_i = datai['obstacle']
    route_i = datai['route']
    obstacle_list = obstacle_list + obstacle_i
    route_list = route_list + route_i
    print('length of file:', len(obstacle_i))

print('length of collected file:', len(obstacle_list))
print('Sample obstacle item', obstacle_list[0])
# dataset
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self):

        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'route': route_list,
            # (N, obs_dim)
            'obstacle': obstacle_list
        }
        self.train_data = train_data

    def __len__(self):
        # all possible segments of the dataset
        return len(self.train_data['obstacle'])

    def __getitem__(self, idx):
    
        # get nomralized data using these indices
        train_route = np.array(self.train_data['route'][idx])
        train_obstacle = np.array(self.train_data['obstacle'][idx])
        train_route = np.reshape(train_route, (1,train_route.shape[0],train_route.shape[1]))
        train_obstacle = np.reshape(train_obstacle, (1,train_obstacle.shape[0],train_obstacle.shape[1]))
        train_data_idx = {
            'route':train_route,
            'obstacle': train_obstacle
        }
        return train_data_idx

"""
dataset = TrajectoryDataset()
# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
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
"""
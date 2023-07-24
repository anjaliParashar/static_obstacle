import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import pickle

goal = [200, 50]
start = [10, 240]
nrows = 256
ncols = 256
[x, y] = np.meshgrid (np.arange(ncols), np.arange(nrows))
# Generate some points
def obstacle_gen(rec1, rec2, cir1,cir2):
    obstacle = np.zeros((nrows, ncols))
    # Generate some obstacle
    
    obstacle [rec1[0]:rec1[0]+50, rec1[1]:rec1[1]+100] = True
    obstacle [rec2[0]:rec2[0]+50, rec2[1]:rec2[1]+100] = True

    t = ((x - cir1[0])**2 + (y - cir1[1])**2) < cir1[2]**2
    obstacle[t] = True

    t = ((x - cir2[0])**2 + (y - cir2[1])**2) < cir2[2]**2
    obstacle[t] = True
    
    return obstacle

def potential(obstacle):
    d = bwdist(obstacle==0)
    # Rescale and transform distances
    d2 = (d/100.) + 1
    d0 = 2
    nu = 800
    repulsive = nu*((1./d2 - 1/d0)**2)
    repulsive [d2 > d0] = 0
    xi = 1/700.
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 )
    f = attractive + repulsive
    return f

def GradientBasedPlanner (f, start_coords, end_coords, max_its, obstacle):
    [gy, gx] = np.gradient(-f)

    route = np.vstack( [np.array(start_coords), np.array(start_coords)] )
    for i in range(max_its):
        current_point = route[-1,:]
#         print(sum( abs(current_point-end_coords) ))
        if sum( abs(current_point-end_coords) ) < 5.0:
            print('Reached the goal !')
            break
        ix = int(round( current_point[1] ))
        iy = int(round( current_point[0] ))
        if ix>=256:
            ix=255
        if iy>=256:
            iy = 255
        if ix<=0:
            ix=0
        if iy<=0:
            iy=0
        vx = gx[ix, iy]
        vy = gy[ix, iy]
        dt = 1 / np.linalg.norm([vx, vy])
        next_point = current_point + dt*np.array( [vx, vy] )
        route = np.vstack( [route, next_point] )
    route = route[1:,:] 
    return route
def gplot(obstacle, route):
    # Display 2D configuration space
    plt.imshow(1-obstacle, 'gray')
    plt.plot (start[0], start[1], 'ro', markersize=10)
    plt.plot (goal[0], goal[1], 'ro', color='green', markersize=10)
    plt.plot(route[:,0], route[:,1], linewidth=0.5)
        # plt.axis ([0, ncols, 0, nrows]);

    plt.xlabel ('x')
    plt.ylabel ('y')

    plt.title ('Configuration Space')

def coordinate_gen():
    rect_10 = np.random.randint(low=100, high = 200)
    rect_11 = np.random.randint(low=100, high = 200)
    rect_20 = np.random.randint(low=150, high = 200)
    rect_21 = np.random.randint(low=50, high = 100)
    rec1 = np.array([rect_10, rect_11])
    rec2 = np.array([rect_20, rect_21])
    cir13 = np.random.randint(low=20, high = 50)
    cir23 = np.random.randint(low=20, high = 50)
    cir10 = np.random.randint(low=0, high = 100)
    cir11 = np.random.randint(low=0, high = 100)
    cir20 = np.random.randint(low=150, high = 200)
    cir21 = np.random.randint(low=150, high = 200)
    cir1 = np.array([cir10,cir11,cir13])
    cir2 = np.array([cir20,cir21,cir23])
    return rec1,rec2, cir1, cir2

route_list = []
obstacle_list = []

N_data = 3000
for i in range(N_data):
    rec1,rec2, cir1, cir2 = coordinate_gen()
    obstacle = obstacle_gen(rec1,rec2, cir1, cir2)
    f = potential(obstacle=obstacle)
    route = GradientBasedPlanner(f, start, goal, 700, obstacle)
    length = route.shape[0]
    padding = np.tile(route[-1,:], (701-length, 1)) 
    route = np.vstack( [route, padding] )
    if sum(abs(route[-1,:]-goal))<10.0:
        route_list.append(route)
        obstacle_list.append(obstacle)
        gplot(obstacle,route)
    if i%300==0:
        print('Check')

data = {"route":route_list, "obstacle":obstacle_list}
print('length of data:', len(route_list))
f_data = "train_data/file_1.pkl"
with open(f_data, 'wb') as fp:
  pickle.dump(data, fp)
fp.close()
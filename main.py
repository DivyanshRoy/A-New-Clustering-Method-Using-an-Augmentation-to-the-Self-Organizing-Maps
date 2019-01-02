from matplotlib import pyplot as plt
import numpy as np
from modified_som import Modified_SOM
from numpy import genfromtxt

data = genfromtxt('data.csv', delimiter=',')

normed_data = (data) / (data.max(axis=0))

no_of_iterations = 1399
som = Modified_SOM(100, 1, 2,normed_data, no_of_iterations)
#print(normed_data.shape)
som.train(normed_data)

mapped = som.map_vects(normed_data,normed_data)
mapped=np.array(mapped)

#print(mapped)


c_grid = som.get_centroids()
c_grid = c_grid*data.max(axis=0)
print(c_grid)
c_grid=np.array(c_grid)
print(c_grid.shape)
centroid_grid=np.zeros((0,2))
i=0
for i in range(c_grid.shape[0]):
    centroid_grid = np.insert(centroid_grid,i,np.array((c_grid[i,0,0],c_grid[i,0,1])),0)
sum_coordinates=np.zeros((c_grid.shape[0],3))

for i in range(mapped.shape[0]):
    cluster_no = np.array(mapped[i,2])
    cluster_no = cluster_no.astype(int)
    sum_coordinates[cluster_no,0] = sum_coordinates[cluster_no,0] + mapped[i,0]
    sum_coordinates[cluster_no,1] = sum_coordinates[cluster_no,1] + mapped[i,1]
    sum_coordinates[cluster_no,2] = sum_coordinates[cluster_no,2] + 1

for i in range(sum_coordinates.shape[0]):
    sum_coordinates[i,0] = sum_coordinates[i,0] / sum_coordinates[i,2]
    sum_coordinates[i,1] = sum_coordinates[i,1] / sum_coordinates[i,2]

for i in range(mapped.shape[0]):
    mn_val = 100000.0
    index3 = -1
    for j in range(sum_coordinates.shape[0]):
        tmp_val = np.sqrt( (mapped[i,0]-sum_coordinates[j,0])*(mapped[i,0]-sum_coordinates[j,0]) + (mapped[i,1]-sum_coordinates[j,1])*(mapped[i,1]-sum_coordinates[j,1]) )
        if tmp_val < mn_val:
            mn_val = tmp_val
            index3 = j;
    index4 = np.array(index3)
    index4 = index4.astype(float)
    mapped[i,2] = index4

for i in range(data.shape[0]):
    plt.scatter(mapped[i,1]*data.max(axis=0)[1],data.max(axis=0)[0]-mapped[i,0]*data.max(axis=0)[0],s=None,c=plt.cm.RdYlBu((mapped[i,2]/20)+0.04))

print(mapped.shape[0])

for mappe in mapped:
    print(mappe[0]*data.max(axis=0)[0],' ',mappe[1]*data.max(axis=0)[1],' ',mappe[2]+1)


plt.show()
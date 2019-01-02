import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
from numpy import genfromtxt

a = genfromtxt('data.csv', delimiter=',')

fig, axes23 = plt.subplots(1, 1)

for axes in [axes23]:
    #z = hac.linkage(a, method='single')
    ##OTHER AVAILABLE METHODS
    z = hac.linkage(a, method='complete')
    #z = hac.linkage(a, method='average')
    #z = hac.linkage(a, method='weighted')
    #z = hac.linkage(a, method='centroid')
    #z = hac.linkage(a, method='median')
    #z = hac.linkage(a, method='ward')

    num_clust1 = 100

    part1 = hac.fcluster(z, num_clust1, 'maxclust')
    part1 = np.array(part1)

    clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    '#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']

    print(part1.shape[0])
    for i in range(part1.shape[0]):
        plt.scatter(a[i,0], a[i, 1],c=plt.cm.RdYlBu((part1[i]/20)+0.04))
        print(a[i,0],' ',a[i,1],' ',part1[i])
    plt.setp(axes, title='{} Clusters'.format(num_clust1))

plt.show()
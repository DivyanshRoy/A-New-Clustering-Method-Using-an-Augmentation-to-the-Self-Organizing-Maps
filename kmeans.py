import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import genfromtxt

data = genfromtxt('data.csv', delimiter=',')
df=pd.DataFrame({'x':data[:,0],'y':data[:,1]})

xval=np.array(df['x'])
yval=np.array(df['y'])

kmeans = KMeans(n_clusters=100)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)
lbl=np.array(labels)

print(xval.shape[0])
for i in range(xval.shape[0]):
    plt.scatter(xval[i],yval.max(axis=0)-yval[i],c=plt.cm.RdYlBu((lbl[i]/20)+0.04))
    print(xval[i],' ',yval[i],' ',lbl[i]+1)

plt.show()
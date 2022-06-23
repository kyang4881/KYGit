from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
print(X)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
print("Labels:")
print(labels)

pred = kmeans.predict([[0, 0], [12, 3]])
print("Pred:")
print(pred)

center = kmeans.cluster_centers_
print("Center:")
print(center)

x = X.ravel(order='F')[0:np.shape(X)[0]]
y = X.ravel(order='F')[np.shape(X)[0]:]

plt.scatter(x,y)
plt.show()

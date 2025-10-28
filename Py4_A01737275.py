import csv
import numpy as np
import copy
import matplotlib.pyplot as plt 
from functools import partial
import random
import math
from matplotlib.image import imread

def euclideanDistance(list1, list2):
    sumList=0
    for x, y in zip(list1, list2):
      sumList=sumList+((y - x) ** 2)
    return math.sqrt(sumList)

def lector_csv(nombre_archivo):
    x = []
    with open(nombre_archivo, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if len(row) > 0:
                x.append([float(y) for y in row[1:]])
    return x

def kMeansInitCentroids(X, K):
  centroids = []
  for i in range(K):
      index = random.randrange(len(X)-1)
      centroids.append(X[index])
  return centroids  

def findClosestCentroids(X, initial_centroids):
    idx = []
    for i in X:
        c_distance = [euclideanDistance(x,i) for x in initial_centroids]
        idx.append(c_distance.index(min(c_distance)))
    return idx

def computeCentroids(X, idx, K):
    features = len(X[0])
    acc = [[0]*features for _ in range(K)]
    count = [0 for _ in range(K)]
    for id,i in zip(idx,X):
        count[id] += 1
        for f in range(features):
            acc[id][f] += i[f]
    for i in range(len(acc)):
        for f in range(features):
            acc[i][f] = acc[i][f]/count[i]
        
    return acc

def runkMeans(X, initial_centroids, max_iters):
    centroids = initial_centroids
    for i in range(max_iters):
        list = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,list,len(centroids))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  
    for k in range(len(centroids)):
        cluster_points = [X[j] for j in range(len(X)) if list[j] == k]
        cluster_points = np.array(cluster_points)
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:,0], cluster_points[:,1], 
                        c=colors[k % len(colors)], label=f'Cluster {k+1}')
    centroids = np.array(centroids)
    plt.scatter(centroids[:,0], centroids[:,1], 
                c='black', marker='X', s=200, label='Centroides finales')
    
    plt.title("K-Means Resultados")
    plt.legend()
    plt.grid(True)
    plt.show()
    return centroids, list


if __name__ == "__main__":
  #------------data
  x = lector_csv("ex7data2.txt")
  centroids = kMeansInitCentroids(x,3)
  runkMeans(x, centroids, 10)
  
  
  #---------imagen
  A = imread("bird_small.png").astype(float)
  if A.max() <= 1.0:
      A = (A * 255).astype(np.uint8)
  X = A.reshape(-1, 3)
  k = 16
  initial_centroids = kMeansInitCentroids(X, k)
  centroids, idx = runkMeans(X, initial_centroids, 10)
  X_compressed = np.array([centroids[i] for i in idx])
  X_compressed = X_compressed.reshape(A.shape) 
  fig, ax = plt.subplots(1, 2, figsize=(8, 4))
  ax[0].imshow(A)
  ax[0].set_title(" imagen Original")
  ax[1].imshow(X_compressed.astype('uint8'))
  ax[1].set_title(f"imagen comprimida con {k} colores")
  plt.show()


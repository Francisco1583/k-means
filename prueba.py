import csv
import numpy as np
import copy
import matplotlib.pyplot as plt 
from functools import partial
import random
import math
from matplotlib.image import imread
from Py4_A01737275 import euclideanDistance, lector_csv, kMeansInitCentroids, findClosestCentroids, computeCentroids, runkMeans

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

import numpy as np
import matplotlib.pyplot as plt 

def visualize_points(X, y):
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.show()

def generateDataPoints(number_of_classes, points_per_class):
  N = points_per_class
  K = number_of_classes
  X = np.zeros((N*K,2)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in np.arange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*(K+1),(j+1)*(K+1),N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
  return X, y

from tsne import bh_sne
import csv
import numpy as np
import _pickle
from tsne import bh_sne
import matplotlib.pyplot as plt
#open('sample_mnist.csv','r','/Users/jiadao/Desktop') as mnist_example
mnist_example = open('/Users/jiadao/Desktop/sample_mnist.csv','rb')
train, val, test = mnist_example
mnist_example.close()
X = np.asarray(np.vstack((train[0], val[0], test[0])), dtype=np.float64)
y = np.hstack((train[1], val[1], test[1]))
X_2d = bh_sne(X)
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)

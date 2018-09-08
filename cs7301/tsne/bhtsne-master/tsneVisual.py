'''
@author: yifan
'''

import numpy as np
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from sklearn import manifold
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import bhtsne


# initialization:
n_neighbors = 30


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i]/10.))
        # plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.))

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)

    

'''
# generating the sample mnist dataset

mnist_path = '/home/yifan/Documents/Datasets/Temp//mldata/mnist-original.mat'


mnist_raw = loadmat(mnist_path)

X = np.array(mnist_raw['data'].T)
mnist_labels = np.array(mnist_raw['label'].T)

mnist_with_label = np.hstack((X, mnist_labels))

# print('X shape is: {}'.format(X.shape))
# print('Label shape is: {}'.format(mnist_labels.shape))
# print('Combined shape is: {}'.format(mnist_with_label.shape))


sample_mnist = mnist_with_label[np.random.randint(mnist_with_label.shape[0], size=3500), :]

# np.savetxt('sample_mnist.csv', sample_mnist, delimiter=',')

# print(sample_mnist.shape)

'''

# mnist_path = '/home/yifan/Dropbox/workspace/AAAI2019/sample_mnist.csv'
# sample_mnist = np.loadtxt('sample_mnist.csv', delimiter=',')
sample_mnist = np.loadtxt('sample_fashionmnist.csv', delimiter=',')
# print(sample_mnist.shape)

X = sample_mnist[:,:-1]
y = sample_mnist[:,-1]

# print(X.shape, y.shape)



#--------------------------------------------------------------------------------------------
# TSNE embedding

print('Now I\'m executing TSNE...')

t0 = time()
embSpace = manifold.TSNE(n_components=2).fit_transform(X)

print('Now I\'m generating figures...')

plot_embedding(embSpace, y, "tSNE embedding, time spent: {}".format(time()-t0))

#--------------------------------------------------------------------------------------------
# Isomap projection

print('Now I\'m executing Isomap...')

t0 = time()
isoSpace = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)

print('Now I\'m generating figures...')

plot_embedding(isoSpace, y, 'Isomap embedding, time spent: {}'.format(time()-t0))


#--------------------------------------------------------------------------------------------
# LLE projection (locally linear embedding)

print('Now I\'m executing LLE...')

t0 = time()
lleSpace = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard').fit_transform(X)

print('Now I\'m generating figures...')

plot_embedding(lleSpace, y, 'LLE projection, time spent: {}'.format(time()-t0))



# #--------------------------------------------------------------------------------------------
# # LLE projection (locally linear embedding)
#
# # print('Now I\'m executing tree-based TSNE')
#
# print('executing!')
#
# t0 = time()
# treeTSNE = bhtsne.run_bh_tsne(X)
#
#
# # print('Now I\'m generating figures...')
#
# plot_embedding(treeTSNE, y, 'Tree-based TSNE, time spent: {}'.format(time()-t0))


plt.show()

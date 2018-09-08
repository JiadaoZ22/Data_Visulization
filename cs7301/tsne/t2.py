import numpy as np
import bhtsne
from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np
import os, sys
import io



mnist_path = '/Users/jiadao/PycharmProjects/Py3/data visulization/tsne/bhtsne-master/mnist2500_X.txt'
data = np.loadtxt(mnist_path, skiprows=1)


print('FINISHED LOADING')
embedding_array = bhtsne.run_bh_tsne(data)
from time import time

import numpy as np

from numpy import array, dot, exp, sqrt
from numpy.random import rand, seed
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

if __name__ == '__main__':
    # load data
    fState = "data/pop_state.csv"
    fLocs = "data/pop_locs.csv"
    fData = "data/pop_data.csv"

    state = np.loadtxt(fState)
    locs = np.loadtxt(fLocs)
    data = np.loadtxt(fData)

    # add figure
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    panels = locs[state == 1]
    empty = locs[state == 0]
    new = locs[state == 2]

    ax.scatter(panels[:,0], panels[:,1], color='green', s=7, alpha=0.5)
    ax.scatter(empty[:,0], empty[:,1], color='red', s=7, alpha=0.5)
    ax.scatter(new[:,0], new[:,1], color='blue', s=7, alpha=0.5)
    plt.title("Final state (GeoData)", size=22)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # diffusion proces
    # fig = plt.figure(figsize=(10, 8))
    # plt.plot(densities)
    # plt.title("Diffusion process", size=22)
    # plt.xlabel("# Updates")
    # plt.ylabel("Density")
    # plt.show()

    fig = plt.figure(figsize=(10, 8))
    ndata = data.reshape((int(data.shape[0] / (2 * 9)), 9, 2))
    #data = np.log(data)
    for k, update in enumerate(ndata[2::], start=0):
        plt.plot(np.log(update[:, 0]), np.log(update[:, 1]), 'o', label="Update # %d" % (2*k+3))
    plt.axis('equal')
    plt.ylabel(" log n")
    plt.xlabel("log r")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.show()

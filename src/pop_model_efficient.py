# Population Model
from time import time

import numpy as np
import os
import sys

from numpy import array, dot, exp, sqrt
from numpy.random import rand, seed
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

import matplotlib.pyplot as plt

import pandas as pd


def density(state): # state: boolean 1-D array
    return  float(sum(state != 0)) / state.size


def calc_p(dmin, method):
    scale = 0.2 # results in n=1.x after 32 iterations (sparse)
    if method == "exp":
        return scale * np.exp(-dmin)
    elif method == "1overr":
        return scale * 1.0/(16 * dmin)
    elif method == "1overr2":
        return scale * 1.0/((8. * dmin * dmin))



def update(locs, state, method):
    panels = locs[state != 0]
    for i, micro_state in enumerate(state):
        if micro_state == 2: # was created in last iteration
            state[i] = 1
        elif micro_state == 0:
            loc = np.asarray([locs[i]])
            dmin = np.min(distance.cdist(loc, panels))
            #dmin = np.min(all_dist[i][state == 1])
            p = calc_p(dmin, method)
            r = rand()
            state[i] = 2 * int(p > r) # 2 if True, 0 else
            #if state[i] == 2: print(dmin, p, r, locs[i])
    n_new = sum(state==2) 
    return n_new



def BC(state, L):
    Lx = L
    Ly = L
    #n = np.sum(state != 0) / state.size

    x0 = 2*np.exp(1)
    x1 = np.exp(1)

    data = locs[state != 0]
            
    scale = x0
    H, edges=np.histogramdd(data, bins=(np.arange(0,Lx+scale/2,scale),np.arange(0,Ly+scale/2,scale)))
    count0 = np.sum(H>0)
    
    scale = x1
    H, edges=np.histogramdd(data, bins=(np.arange(0,Lx+scale/2,scale),np.arange(0,Ly+scale/2,scale)))
    count1 = np.sum(H>0)
    
    dx = np.log(1./x1) - np.log(1./x0)
    dy = np.log(count1) - np.log(count0)
    return dy / dx


if __name__ == '__main__':

    seed(1)

    # initial panel density
    # at start of simulation
    # the reference density for the analysis is n=3%
    f = int(sys.argv[1])
    N = 1000 * f
    L = 20 * sqrt(f)
    n0 = 0.0001

    locs = rand(N, 2) * L
    #all_dist = squareform(pdist(locs))

    # initial state
    tol = 0.1 # 10 percent
    state = 2 * np.array(rand(N) < n0, dtype='int32')
    n_real = density(state)
    abs_tol = n0 * tol
    is_valid_n = lambda _n: _n < n0 + abs_tol and _n > n0 - abs_tol
    while not is_valid_n(n_real):
        state = 2 * np.array(rand(N) < n0, dtype='int32')
        n_real = density(state)

    print("Real inital desity n_i = %.4f" % n_real)

    # save densities for visualizing global diffusion
    densities = []

    #----------------
    # run simulation
    #----------------
    method = sys.argv[2]
    dir = "data/sim{}k/history-sparse/{}/".format(f, method)
    # dir = "data/sim{}k/history/".format(f)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    fState = dir + "state_initial.csv"
    fLocs = dir + "locs.csv"
    np.savetxt(fState, state, fmt="%d")
    np.savetxt(fLocs, locs)

    fBC = dir + "BC.txt"

    save_densities = [0.6, 0.2, 0.15] #- stol
    stol = 0.1
    n_steps = 50 # MC steps
    for step in range(n_steps):
        n_new = update(locs, state, method)
        n = density(state)
        print("update # %d\t n=%.4f" % (step+1, n))

        # save state
        for ni in save_densities:
            if abs(n - ni) < stol:
                fState = dir + "state_iter_%d.csv" % (step+1)
                np.savetxt(fState, state, fmt="%d")
                save_densities.remove(ni)

        # BC
        count = BC(state, L)
        with open(fBC, "a+") as file:
            file.write("%d %.5f %.6f\n" % (step, n, count))

    n_final = density(state)
    print("Real final density n = %.4f\n" % n_final)

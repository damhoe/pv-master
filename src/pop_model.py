# Population Model
from time import time

import numpy as np

from numpy import array, dot, exp, sqrt
from numpy.random import rand, seed
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd


def density(state): # state: boolean 1-D array
    int_state = array(state, dtype='int32')
    return  float(sum(int_state)) / N


def update(locs, state):
    panels = locs[state==1]
    for i, panel in enumerate(state):
        if panel == 2: # was created in last iteration
            state[i] = 1
        elif panel == 0:
            loc = np.asarray([locs[i]])
            dmin = np.min(distance.cdist(loc, panels))
            #dmin = np.min(all_dist[i][state == 1])
            scale = 0.2
            p = scale * np.exp(-dmin)
            r = rand()
            state[i] = 2 * int(p > r) # 2 if True, 0 else
            #if state[i] == 2: print(dmin, p, r, locs[i])
    print("# new panels  Np = %d\n" % sum(state==2))


def count_panels_at_fixed_distance(dmin, dmax, distance_matrix, state):
    #tol = 1. # km
    #if d < tol: raise Exception("Negative distance encountered.")
    #dmin = d - tol
    #dmax = d + tol

    #d_array = distance.cdist(np.asarray(p0), panels)[0]
    #helper = all_dist[state==1]
    #tot_count = np.sum(np.logical_and(helper < dmax, helper > dmin), axis=1)
    count = np.sum(np.logical_and(distance_matrix < dmax, distance_matrix > dmin), axis=1)
    return  float(np.mean(count)) / (dmin * dmin)


if __name__ == '__main__':

    seed(1)

    # initial panel density
    # at start of simulation
    # the reference density for the analysis is n=3%
    f = 10
    N = 1000 * f
    L = 20 * sqrt(f)
    n0 = 0.005

    locs = rand(N, 2) * L
    #all_dist = squareform(pdist(locs))

    # initial state
    state = np.array(rand(N) < n0, dtype='int32')

    n_real = density(state)
    print("Real inital desity n_i = %.4f\n" % n_real)

    # save densities for visualizing global diffusion
    densities = []

    #----------------
    # run simulation
    #----------------
    n_steps = 10 # MC steps

    tol = 1e-3
    radii = np.logspace(np.log10(sqrt(N)/L), np.log10(L/4), 10, endpoint=True)
    data = np.array([], dtype='float64')

    tStart = time()
    for step in range(n_steps):
        update(locs, state)
        n = density(state)
        #densities.append(n)
        print("update # %d\t n=%.3f\n" % (step, n))

        # evaluation
        # ----------------------------------------------------------------
        centers = locs[state==1]
        new_panels = locs[state==2]
        distance_matrix = distance.cdist(centers, new_panels)
        for k in range(len(radii) - 1):
            count = count_panels_at_fixed_distance(radii[k], radii[k+1], distance_matrix, state)
            data = np.append(data, [radii[k], count / sum(state==2)])

    fState = "data/pop_state.csv"
    fLocs = "data/pop_locs.csv"
    fData = "data/pop_data.csv"

    np.savetxt(fState, state, fmt="%d")
    np.savetxt(fLocs, locs)
    np.savetxt(fData, data)

    elapsed = time() - tStart
    print("Elapsed time %f" % elapsed)
    n_final = density(state)
    print("Real final density n = %.4f\n" % n_final)

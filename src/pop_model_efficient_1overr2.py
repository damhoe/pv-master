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
            scale = 0.04
            p = scale * 1.0/((4.8 * dmin)*(4.8 * dmin)) #np.exp(-dmin)
            r = rand()
            state[i] = 2 * int(p > r) # 2 if True, 0 else
            #if state[i] == 2: print(dmin, p, r, locs[i])
    print("# new panels  Np = %d\n" % sum(state==2))


def count_panels_at_fixed_distance(dmin, dmax, new_panels, panels):
    #tol = 1. # km
    #if d < tol: raise Exception("Negative distance encountered.")
    #dmin = d - tol
    #dmax = d + tol

    #d_array = distance.cdist(np.asarray(p0), panels)[0]
    #helper = all_dist[state==1]
    #tot_count = np.sum(np.logical_and(helper < dmax, helper > dmin), axis=1)
    count = np.asarray([])
    for p in panels:
        M1 = distance.cdist(np.asarray([p]), new_panels) # Mem scales as N
        c = np.sum(np.logical_and(M1 > dmin, M1 < dmax), axis=1)
        count = np.append(count, c[0])
    
    return  float(np.mean(count)) / (dmin * dmin)


def count_panels_with_fixed_dmin(dmin, dmax, new_panels, panels, locs):
    # distance.cdist(new_panels, panels) scales with N^2
    # which let mem rsc increase immensly
    # alternitive: use for loop
    # old:
    # ----
    #DM1 = distance.cdist(new_panels, panels)
    #DM2 = distance.cdist(new_panels, locs)
    #count = np.sum(np.logical_and(DM1 > dmin, DM1 < dmax), axis=1)
    #tot_count = np.sum(np.logical_and(DM2 > dmin, DM2 < dmax), axis=1)
    # new:
    # ----
    count = np.asarray([])
    tot_count = np.asarray([])
    for p in new_panels:
        M1 = distance.cdist(np.asarray([p]), panels) # Mem scales as N
        M2 = distance.cdist(np.asarray([p]), locs)
        c = np.sum(np.logical_and(M1 > dmin, M1 < dmax), axis=1)
        tc = np.sum(np.logical_and(M2 > dmin, M2 < dmax), axis=1)
        count = np.append(count, c[0])
        tot_count = np.append(tot_count, tc[0])

    return np.mean(count / tot_count)


if __name__ == '__main__':

    seed(1)

    # initial panel density
    # at start of simulation
    # the reference density for the analysis is n=3%
    f = 1000
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
    n_steps = 20 # MC steps

    tol = 1e-3
    radii = np.logspace(np.log10(sqrt(N)/L), np.log10(L/4), 10, endpoint=True)
    nr_fixed_distance = np.array([], dtype='float64')
    nr_fixed_dmin = np.array([], dtype='float64')

    tStart = time()
    for step in range(n_steps):
        update(locs, state)
        n = density(state)
        #densities.append(n)
        print("update # %d\t n=%.3f\n" % (step, n))

        panels = locs[state==1]
        new_panels = locs[state==2]
    
        # evaluation 1
        # ----------------------------------------------------------------
        for k in range(len(radii) - 1):
            count = count_panels_with_fixed_dmin(radii[k], radii[k+1], new_panels, panels, locs)
            nr_fixed_dmin = np.append(nr_fixed_dmin, [radii[k], count / sum(state==2)])
            
        # evaluation 2
        # ----------------------------------------------------------------
        for k in range(len(radii) - 1):
            count = count_panels_at_fixed_distance(radii[k], radii[k+1], new_panels, panels)
            nr_fixed_distance = np.append(nr_fixed_distance, [radii[k], count / sum(state==2)])

    fState = "data/sim1000k/pop_state_1overr2.csv"
    fLocs = "data/sim1000k/pop_locs_1overr2.csv"
    fData1 = "data/sim1000k/pop_data_eval1_1overr2.csv"
    fData2 = "data/sim1000k/pop_data_eval2_1overr2.csv"

    np.savetxt(fState, state, fmt="%d")
    np.savetxt(fLocs, locs)
    np.savetxt(fData2, nr_fixed_dmin)
    np.savetxt(fData1, nr_fixed_distance)

    elapsed = time() - tStart
    print("Elapsed time %f" % elapsed)
    n_final = density(state)
    print("Real final density n = %.4f\n" % n_final)

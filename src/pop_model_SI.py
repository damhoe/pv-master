"""Single Island Population Model

Here multiple settings are simulated with each having
only one island at t=0 somewhere in the center of the area.

"""

# Population Model
from cProfile import run
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
from numba import njit


def density(state): # state: boolean 1-D array
    return  float(sum(state != 0)) / state.size


def calc_p(dmin, method):
    #scale = 0.2 # results in n=1.x after 32 iterations (sparse)
    if method == "exp":
        scale = 0.2
        return scale * np.exp(-dmin)
    elif method == "1overr":
        scale = 1e-3
        #limit = 1e1
        #if dmin > limit:
        #    return 0
        return scale * 1.0 / dmin
    elif method == "1overr2":
        scale = 1
        limit = 1e1
        if dmin > limit:
            return 0
        return scale * 1.0/(dmin * dmin)
    elif method == 'random':
        return 1. / (0.44 * 50)



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


def get_limit(lower, upper, box_size):
    d = upper - lower
    rest = d % box_size
    # needs to be above stop
    # because of np.arange([start, stop)) open interval
    overhang = box_size * 1e-1
    return [lower, upper + box_size - rest + overhang]
    

def count_boxes(points, box_size, limits=None):
    # get limits in each dimension
    if not limits:
        limits = np.asarray([[np.amin(coords_i), np.amax(coords_i)]for coords_i in np.transpose(points)])
    # modify upper limits based on box_side
    limits = [get_limit(lower, upper, box_size) for lower, upper in limits]    
    H, edges = np.histogramdd(points, bins=tuple(np.arange(start, stop, box_size) \
                                            for start, stop in limits))
    count = np.sum(H > 0)
    return count

def BC(locs, state, L):
    cells = locs[state != 0]
    
    dist = distance.squareform(distance.pdist(cells))
    dist[dist < 1e-7] = 100.
    m = np.min(dist, axis=0)
    xmin = np.mean(m)
    
    x0 = np.log10(4*xmin)
    x1 = np.log10(8*xmin)
    scales = np.logspace(x0, x1, 5)

    helper = lambda scale: count_boxes(cells, scale)
    count_boxes_vect = np.vectorize(helper)
    counts = count_boxes_vect(scales)

    m, y0 = np.polyfit(np.log10(scales[:]), np.log10(counts[:]), deg=1)
    print(m)

    #dx = np.log(1./x1) - np.log(1./x0)
    #dy = np.log(count1) - np.log(count0)
    return abs(m)


def grass_procaccia(N, locs, r):
    # Algorithm
    # ---------
    # Corr(r) = 2 / N (N-1) * Sum_i<j (Theta( r - d_ij ))
    if N > 15000:
        raise Exception('Potential Mem Overload!')
    
    pair_distances = pdist(locs)
    count = np.sum(np.heaviside(r - pair_distances, np.ones(int(N*(N-1)/2))))
    return 2.0 * count / (N*(N - 1))


def GP(locs, state, L):
    r1 = 10**(-1.0)
    r2 = 1.0
    cells = locs[state != 0]
    c1 = grass_procaccia(cells.shape[0], cells, r1)
    c2 = grass_procaccia(cells.shape[0], cells, r2)
    dr = np.log10(r1) - np.log10(r2)
    dc = np.log10(c1) - np.log10(c2)
    return dc / dr


def check_point_inside_circle(px, py, cx, cy, r):
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy < r * r


def select_rnd_center(locs, L):
    """ The center area is defined as a circle
    with radius x percent of L
    """
    centers = []
    p = 0
    while len(centers) < 1:
        p += 1
        r = p * 0.01 * L
        centers = [l for l in locs if check_point_inside_circle(l[0], l[1], L/2, L/2, r)]

    index = np.random.choice(np.arange(len(centers)))
    return centers[index]


@njit
def index(arr2d, val):
    for i, elem in enumerate(arr2d):
        if np.array_equal(val, elem):
            return i
    return None


def run_sim(sim_count):
    seed(sim_count)

    # initial panel density
    # at start of simulation
    # the reference density for the analysis is n=3%
    
    N = int(1e5)
    L = 20 * sqrt(100)

    locs = rand(N, 2) * L

    # initial state
    # 1 initial cell
    state = np.zeros(N)
    center = select_rnd_center(locs, L)
    state[index(locs, center)] = 1

    # save densities for visualizing global diffusion
    densities = []

    #----------------
    # run simulation
    #----------------
    method = '1overr'#sys.argv[2]
    dir = "data/SI/big2/sim{}/{}/".format(sim_count, method)

    if not os.path.isdir(dir):
        os.makedirs(dir)

    #fState = dir + "state_initial.csv"
    fLocs = dir + "locs.csv"
    #np.savetxt(fState, state, fmt="%d")
    np.savetxt(fLocs, locs)

    #fBC = dir + "BC.txt"
    # clear file
    #with open(fBC, "w+") as file:
    #    file.write("")

    fGP = dir + "grass_proccacia.txt"
    # clear file
    with open(fGP, "w+") as file:
        file.write("")

    save_densities = [0.6, 0.2, 0.15] #- stol
    stol = 0.1
    n_steps = 31 # MC steps
    for step in range(n_steps):
        n_new = update(locs, state, method)
        n = density(state)
        #print("update # %d\t n=%.4f" % (step+1, n))

        #save state
        #for ni in save_densities:
        #    if abs(n - ni) < stol:
        if step % 5 == 0:
            fState = dir + "state_iter_%d.csv" % (step+1)
            np.savetxt(fState, state, fmt="%d")
            #save_densities.remove(ni)

        # Fractal Dimension
        #dim_bc = BC(locs, state, L)
        #with open(fBC, "a+") as file:
        #    file.write("%d %.5f %.6f\n" % (step, n, dim_bc))

        dim_gp = GP(locs, state, L)
        #print(f'GP fractal dim: {dim_gp}\n')
        with open(fGP, "a+") as file:
            file.write("%d %.5f %.6f\n" % (step, n, dim_gp))

    n_final = density(state)
    print("Real final density n = %.4f\n" % n_final)


if __name__ == '__main__':

    n_sim = 1#00
    for i in range(n_sim):
        run_sim(i)

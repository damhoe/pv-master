"""
Solar module distribution simulation.

"""


"""Analysis framework for checking samples from probability distributions

(1) Features (including proximity value) are calculated
(2) Classifiers are trained (RF)
(3) Permutation feature importance is calculated and fitted with an exponential function.

"""
from time import time
from datetime import date, datetime

import numpy as np
from numpy import array, linspace, tile, arange, \
                    count_nonzero, zeros, divide, \
                    logical_and, copy, stack, \
                    dot, exp, sqrt, inf
from numpy.random import rand, seed
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize, Bounds, curve_fit

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd

def init_features(df, N, state, dists, radii):
    """
    Calculate density for given area with radius r.

    """

    # helper for calculating density gains
    mState = tile(state, (N, 1))
    previous_density = zeros(N)

    for radius in radii:
        loc_in_range = dists < radius
        mCells = logical_and(mState, loc_in_range)

        n_cells = count_nonzero(mCells, axis=1)
        n_locs = count_nonzero(loc_in_range, axis=1) # symmetric, which axis doesn't matter
        density = divide(n_cells, n_locs)

        density_gain = density - previous_density

        df["gain-r-%.2f" % radius] = density_gain
        df["r-%.2f" % radius] = density

        previous_density = density

    return df

def mc_step(exp_all_dist, locs, state, N):
    #
    # iterate over all locations
    # and update with weight exp(-d_r)

    rnds = rand(N) * N

    for i, panel in enumerate(state):
        if not panel:
            p = dot(exp_all_dist[i], state)
            state[i] = p > rnds[i]

# all distances are normalized by r0 = 0.21 km
# i.e. r=20 corresponds to r_real = 210 m * 20 = 4.2 km
#
# $ Fresno area (about 270 km^2)
# -> model area 256 km^2 (= 16 x 16 km^2)

# original number of addresses N=3*10^5
# rescale for computability (limited memory capacity)
def main():

    r0 = 0.21 # km
    scale = 1.0 / 10

    NO = 300000
    N = int(NO * scale)

    # area under test L^2
    # rescale by sqrt(scale)
    L = 16.0 / r0 * sqrt(scale) # km

    # initial panel density at start of simulation
    # the reference density for the analysis is n=3%
    n0 = 0.01

    seed(0)

    # create pseudo-random locations
    locs = rand(N, 2) * L
    all_dist = squareform(np.asarray(pdist(locs), dtype='float32'))

    # initial state
    state = rand(N) < n0

    PanelDensity = lambda state: sum(array(state, dtype='int32')) * 1.0 / N
    n_real = PanelDensity(state)

    exp_all_dist = exp(-all_dist)

    # save some populations and densities
    # for visualizing and further analysis
    densities = []

    #----------------
    # run simulation
    #----------------
    n_steps = 1000
    tStart = time()

    tol = 1e-3
    ns = [0.02, 0.03, 0.05, 0.7, 0.1, 0.2]

    # for file names
    today = date.today()
    now = datetime.now()
    KEY = today.strftime("%b-%d-%Y") + "-" + now.strftime("%H-%M-%S")

    for step in range(n_steps):
        mc_step(exp_all_dist, locs, state, N)
        n = PanelDensity(state)
        densities.append(n)

        if len(ns) > 0:
            if abs(n - ns[0]) < tol:
                ns.pop(0)

                df = pd.DataFrame(state, columns=["state"])
                df["locx"] = locs[:, 0]
                df["locy"] = locs[:, 1]

                # init parameters for feature calculation
                rmin = 0.2 / r0
                rmax = 1.2 / r0
                step = 0.1 / r0 # 0.1 km increaments

                radii = arange(rmin, rmax, step)
                df = init_features(df, N, state, all_dist, radii)

                df.to_csv("data/data_" + KEY + "_n-%.2f_" % n + ".csv")
            #state2file(state, locs, n, all_dist)
            #populations.append(Population(state, locs, n, mDist=all_dist))

    elapsed = time() - tStart

    LOG_FILE_NAME = "log/Sim_" + KEY + ".log"

    log_file = open(LOG_FILE_NAME, "w+")
    log_file.write("Execution Time = %f\n" % elapsed)
    log_file.write("\n".join(["%.2f" % item for item in densities]))
    log_file.close()

    return

#=============
#= Main Code =
#=============
if __name__ == '__main__':
    main()
    # END

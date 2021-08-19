"""
Solar module distribution simulation.

"""


"""Analysis framework for checking samples from probability distributions

(1) Features (including proximity value) are calculated
(2) Classifiers are trained (RF)
(3) Permutation feature importance is calculated and fitted with an exponential function.

"""
from time import time

import numpy as np
from numpy import array, linspace, tile, \
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

# constant
r0 = 0.21 # km

class Population:

    def __init__(self, state, locs, n, mDist=array([])):
        self.N = state.size
        self.state = copy(state)
        self.locs = copy(locs)
        self.n = n
        self.features = dict()
        self.features_gain = dict()

        if mDist.shape != (self.N, self.N):
            self.mDist = squareform(pdist(locs))
        else:
            self.mDist = mDist

    def to_file(self, fname):
        with open(fname, "w+") as mFile:
            mFile.write("Population with N=%d and n=%.3f\n" % (self.N, self.density()))
            mFile.write("Count,State,LocactionX,LocationY\n")
            for i in range(self.N):
                mFile.write("%d,%d,%.4f,%.4f\n" % (i, self.state[i], self.locs[i][0], self.locs[i][1]))

    def density(self):
        int_state = array(self.state, dtype='int32')
        return sum(int_state) * 1.0 / self.N

    def show(self):
        # add figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)


        cells = self.locs[self.state == True]
        empty = self.locs[self.state == False]

        ax.scatter(cells[:,0], cells[:,1], color='green', s= 10, alpha=0.5)
        ax.scatter(empty[:,0], empty[:,1], color='red', s=10, alpha=0.5)

        return fig, ax

    def init_features(self, rmin, rmax, nr):
        """
        Calculate density for given area with radius r.

        """
        # init the radii
        self.features = dict()
        self.features_gain = dict()
        radii = linspace(rmin, rmax, num=nr, endpoint=True)

        # helper for calculating density gains
        mState = tile(self.state, (self.N, 1))
        previous_density = zeros(self.N)

        for radius in radii:
            loc_in_range = self.mDist < radius
            mCells = logical_and(mState, loc_in_range)

            n_cells = count_nonzero(mCells, axis=1)
            n_locs = count_nonzero(loc_in_range, axis=1) # symmetric, which axis doesn't matter
            density = divide(n_cells, n_locs)

            density_gain = density - previous_density

            self.features_gain[radius] = density_gain
            self.features[radius] = density

            previous_density = density

    def analyse(self, file):
        """
        Train Random Forest and compute feature importance.
        """

        # prepare data
        data = stack(list(self.features.values()), axis=1)
        targets = self.state

        X_train, X_test, y_train, y_test = train_test_split(
            data, targets, random_state=100)

        n_est = 100
        model = RF(n_estimators=n_est).fit(X_train, y_train)

        # output
        file.write("Trained Random Forest with n_est=%d.\n" % n_est)
        file.write("--- Test Sample Size: \t %d\n" % y_test.size)
        file.write("--- Train Sample Size: \t %d\n" % y_train.size)
        file.write("\n--> Score: \t %.2f\n" % model.score(X_test, y_test))

        # calculate importances
        imp = permutation_importance(
            model, X_test, y_test,
            n_repeats=40, random_state=100, scoring='roc_auc')



        return imp.importances_mean, imp.importances_std

    def analyse_gain(self, file):
        """
        Train Random Forest and compute feature importance
        for density gain features.
        """

        # prepare data
        data = stack(list(self.features_gain.values()), axis=1)
        targets = self.state

        X_train, X_test, y_train, y_test = train_test_split(
            data, targets, random_state=0)

        n_est = 100
        model = RF(n_estimators=n_est).fit(X_train, y_train)

        # output
        file.write("Trained Random Forest with n_est=%d.\n" % n_est)
        file.write("--- Test Sample Size: \t %d\n" % y_test.size)
        file.write("--- Train Sample Size: \t %d\n" % y_train.size)
        file.write("\n--> Score: \t %.2f\n" % model.score(X_test, y_test))

        # calculate importances
        imp = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=0, scoring='roc_auc')



        return imp.importances_mean, imp.importances_std

def mc_step(exp_all_dist, locs, state, N):
    #
    # iterate over all locations
    # and update with weight exp(-d_r)

    rnds = rand(N) * N

    for i, panel in enumerate(state):
        if not panel:
            p = dot(exp_all_dist[i], state)
            state[i] = p > rnds[i]

def model(x, a, r0):
    return a * np.exp(- x / r0)

def run_analysis(p, density_gain=True, fname=""):
    # init parameters for feature calculation
    rmin = 0.2 / r0
    rmax = 1.2 / r0
    nr = int((rmax - rmin) * r0 / 0.1) # 0.1 km increaments
    p.init_features(rmin, rmax, nr)

    mFile = open(fname, "w+")

    if density_gain:
        mFile.write("## Density Gain\n")
        raw_mean, raw_std = p.analyse_gain(mFile)
    else:
        mFile.write("## Absolute Density\n")
        raw_mean, raw_std = p.analyse(mFile)

    mean = raw_mean * 1
    std = raw_std * 1

    # fit exp function to data
    x0 = array([0.0, 0.1])
    bnds = (array([-inf, 0.0]), array([inf, inf]))
    #nmax = 5

    mFile.write("\nRadius,Mean,Std\n")
    for r, m, s in zip(array(list(p.features.keys())[:]), mean, std):
        mFile.write("%.2f,%.4f,%.4f\n" % (r, m, s))

    try:
        popt, pcov = curve_fit(model, array(list(p.features.keys())[:]), mean[:], p0=x0, bounds=bnds)

        # output
        #print("\nSolver terminated successfully with nit=%d" % result.nit)
        perr = np.sqrt(np.diag(pcov))
        mFile.write("\n-----------------------\n")
        mFile.write("Fit: r0,err_r0,a,err_a\n")
        mFile.write("%.4f,%.4f,%.4f,%.4f\n" % (popt[1], perr[1], popt[0], perr[0]))
        #print("--> b = %.3f" % result.x[2])
    except RuntimeError:
        mFile.write('\n Analysis failed -> not converging\n')

def main():
    # ----------------
    # model parameters
    # ----------------
    #
    # all distances are normalized Sby r0 = 0.21 km
    # i.e. r = 20 corresponds to r_real = 210 m * 20 = 4.2 km
    #
    # $ Fresno area (about 270 km^2)
    # -> model area 256 km^2 (= 16 x 16 km^2)


    # number of addresses
    ## original number N=3*10^5
    ## rescale for computability (limited memory capacity)
    NO = 300000
    scale = 1.0 / 25
    N = int(NO * scale)

    # area under test L^2
    # rescale by sqrt(scale)
    L = 16.0 / r0 * sqrt(scale) # km

    # initial panel density
    # at start of simulation
    # the reference density for the analysis is n=3%
    n0 = 0.01

    seed(0)

    # create pseudo-random locations
    locs = rand(N, 2) * L
    all_dist = squareform(pdist(locs))

    # initial state
    state = rand(N) < n0

    PanelDensity = lambda state: sum(array(state, dtype='int32')) * 1.0 / N

    n_real = PanelDensity(state)

    print("Real inital desity n_i = %.4f\n" % n_real)

    exp_all_dist = exp(-all_dist)

    # save some populations and densities
    # for visualizing and further analysis
    densities = []
    populations = []

    #----------------
    # run simulation
    #----------------
    n_steps = 1000
    tStart = time()

    tol = 1e-3
    ns = [0.01]#, 0.02, 0.04, 0.1,]

    for step in range(n_steps):
        mc_step(exp_all_dist, locs, state, N)
        n = PanelDensity(state)
        densities.append(n)

        #if step >= 100 and step < 600 \
        #    and step % 100 == 0:
        for ni in ns:
            if abs(ni - n) < tol:
                ns.pop(0)
                # print information
                print("At step %d -> Population saved with n = %.3f\n" % (step, n))
                populations.append(Population(state, locs, n, mDist=all_dist))

    elapsed = time() - tStart
    print("Elapsed time %f" % elapsed)

    for p in populations:
        p.to_file("new_pop.data")

        # First, calculate the features
        # -> density gain within given radius

        name = "new_test_analysis"
        run_analysis(p, density_gain=True, fname=name + "_gain.data")
        run_analysis(p, density_gain=True, fname=name + ".data")

    return

#=============
#= Main Code =
#=============
if __name__ == '__main__':
    main()
    # END

"""
Calculation of p(x != 0)(dmin).

"""
import numpy as np
from scipy.spatial import distance


def main():
    # load data
    # load data
    #    fState = "data/sim1000k/pop_state.csv"
    #    fLocs = "data/sim1000k/pop_locs.csv"

    fState = "data/sim1000k/pop_state_1overr.csv"
    fLocs = "data/sim1000k/pop_locs_1overr.csv"

    # fState = "data/sim1000k/pop_state_1overr2.csv"
    # fLocs = "data/sim1000k/pop_locs_1overr2.csv"

    state = np.loadtxt(fState)
    locs = np.loadtxt(fLocs)

    # build array [dmin, X]
    #-----------------------
    data = []
    search_radius = 15
    cidx = np.random.choice(np.arange(0, state.size, 1), 50000)
    for i, (location, X) in enumerate(zip(locs[cidx], state[cidx])):
        upper = location + search_radius
        lower = location - search_radius
        panels = locs[state == 1]
        inside_search_area = np.sum(np.logical_and(panels < upper, panels > lower), axis=1) == 2
        near_panels = panels[inside_search_area]
        if len(near_panels) == 0:
            print("Too small search radius!")
            break
        dist_to_panels = distance.cdist(np.asarray([location]), near_panels)
        tol = 1e-4
        dmin = np.min(dist_to_panels[dist_to_panels > tol])
        data.append([dmin, X])

    data = np.asarray(data)

    # save data
    # fData = "data/sim1000k/pop_data_panel_distr.csv"
    fData = "data/sim1000k/pop_data_panel_distr_1overr.csv"
    #fData = "data/sim1000k/pop_data_panel_distr_1overr2.csv"
    np.savetxt(fData, data)



if __name__ == "__main__":
    main()

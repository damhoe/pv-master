import numpy as np

def fractal_eval(state, locs):
    

    # Final density of the state
    print("Final Density:")
    print(np.sum(state >=1) / state.size)

    # computing the fractal dimension
    #considering only scales in a logarithmic list

    data = locs[np.logical_or(state==1, state==2)]

    N = state.size
    L = 20 * sqrt(N / 1000)
    Lx = L
    Ly = L

    start = np.log10(L * 1./2**8)
    end = np.log10(L * 1./2**13)

    scales=np.logspace(start, end, 10)
    Ns=[]
    # looping over several scales
    for scale in scales:
        print ("======= Scale :",scale)
        # computing the histogram
        H, edges=np.histogramdd(data, bins=(np.arange(0,Lx+scale/2,scale),np.arange(0,Ly+scale/2,scale)))
        Ns.append(np.sum(H>0))

    eps = scales
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(eps)[3:6], np.log(Ns)[3:6], 1)

    plt.plot(np.log(eps),np.log(Ns), 'o', mfc='none')
    plt.plot(np.log(eps), np.polyval(coeffs,np.log(eps)))
    plt.xlabel('log $\epsilon$')
    plt.ylabel('log N')
    plt.axis("equal")

    print ("The Hausdorff dimension is", -coeffs[0]) #the fractal dimension is the OPPOSITE of the fitting coefficient

    # plot subarea
    # add figure
    N = state.size
    L = 20 * sqrt(N / 1000)

    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    upper = 330
    lower = 270
    inner_cond = np.prod(np.logical_and(locs < upper, locs > lower), axis=1)

    panels = locs[np.logical_and(inner_cond, state == 1)]
    empty = locs[np.logical_and(inner_cond, state == 0)]
    new = locs[np.logical_and(inner_cond, state == 2)]

    ax.scatter(empty[:,0], empty[:,1], color='red', s=2, alpha=0.3)
    ax.scatter(panels[:,0], panels[:,1], color='green', s=2, alpha=0.9)
    ax.scatter(new[:,0], new[:,1], color='blue', s=2, alpha=0.6)
    plt.title("Final state (GeoData)", size=22)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')

    rectangle1 = plt.Rectangle((300,300), scales[0], scales[0], fc=(0, 0, 0, 0), ec="red")
    rectangle2 = plt.Rectangle((300,300), scales[1], scales[1], fc=(0, 0, 0, 0), ec="red")
    plt.gca().add_patch(rectangle1)
    plt.gca().add_patch(rectangle2)

    plt.show()
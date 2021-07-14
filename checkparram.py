import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def func(x):
    return x**2 + np.cos(x + 2)

def dfunc(x):
    return 2*x - np.sin(x + 2)

def paramshift(func, x, r, phi):
    return r*(func(x + phi) - func(x - phi))

def share(k, b):
    rs = np.linspace(-100,100, k)
    phases = np.linspace(-50,50, b)
    min_fidelity, gradient_of_min_fidelity = 10000, 0
    fidelity = []
    pss, rss = [], []
    for r in rs:
        for p in phases:
            x = 0
            gradient = dfunc(x)
            psuedogradient = paramshift(func, x, r, p)
            
            error = np.abs(gradient - psuedogradient)
            if error < min_fidelity: 
                min_fidelity = error
                gradient_of_min_fidelity = gradient
            fidelity.append(error)
            
            pss.append(p)
            rss.append(r)

    standard_err = min_fidelity
    relative_err = min_fidelity/gradient_of_min_fidelity*100
    return standard_err, relative_err

pss = []
rss = []
err = []
for i in range(2, 30, 1):
    for j in range(2, 30, 1):
        standard_err, relative_err = share(i,j)
        pss.append(i)
        rss.append(j)
        err.append(standard_err)
        print(i)
        print(j)




fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_trisurf(np.asarray(pss), np.asarray(rss), np.asarray(err), cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('nε')
ax.set_ylabel('nR')
ax.set_zlabel('Min fidelity')
plt.show()

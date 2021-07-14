import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def func(x):
    return np.log(x)

def dfunc(x):
    return 1/x

def paramshift(func, x, r, phi):
    return r*(func(x + phi) - func(x - phi))


rs = np.linspace(-100,100, 100)
phases = np.linspace(-50,50, 100)
min_fidelity, gradient_of_min_fidelity, min_r, min_phase = 1000000, 0, 0, 0
fidelity = []
pss, rss = [], []
for r in rs:
    for p in phases:
        x = np.random.normal(10, 5)
        gradient = dfunc(x)
        psuedogradient = paramshift(func, x, r, p)
        
        error = np.abs(gradient - psuedogradient)
        if error < min_fidelity: 
            min_fidelity = error
            gradient_of_min_fidelity = gradient
            min_r = r
            min_phase = p
        fidelity.append(error)
        
        pss.append(p)
        rss.append(r)

print("Min r: " + str(min_r))
print("Min phase: " + str(min_phase))
print(min_fidelity)
print(gradient_of_min_fidelity)
print(str(min_fidelity/gradient_of_min_fidelity*100) + "%")

fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_trisurf(np.asarray(pss), np.asarray(rss), np.asarray(fidelity), cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Phase')
ax.set_ylabel('r')
ax.set_zlabel('Fidelity')
plt.show()

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
min_fidelity, gradient_of_min_fidelity, min_r, min_phase = 10000000000, 0, 0, 0
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
ae = []
re = []
reals = []
fakes = []
for i in range(0, 100):
    x = np.random.normal(10, 5)
    reals.append(dfunc(x))
    fakes.append(paramshift(func, x, min_r, min_phase))
    re.append(np.abs(dfunc(x)-paramshift(func, x, min_r, min_phase))/(dfunc(x)))
plt.plot(range(0,100), reals, label="Exact gradient")
plt.plot(range(0,100), fakes, label="Approximate gradient")
plt.legend(loc='lower left', borderaxespad=0.)
plt.show()
print(min_r)
print(min_phase)
print(min_fidelity)
print(np.average(re))

#-44.53781512605042
#-9.663865546218489

#1.0101010101010104
#0.5050505050505052
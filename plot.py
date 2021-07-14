import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
fig, ax = plt.subplots()
ax.plot(x, (x**2+3*x-1)/(x+1))
ax.set_aspect('equal')
ax.grid(True, which='both')

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.show()
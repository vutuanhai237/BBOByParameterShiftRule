import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import base_psx.base


def func(x):
    return x**2 + np.cos(x + 2)

def dfunc(x):
    return 2*x - np.sin(x + 2)

def func2(x):
    return np.sin(x)

def dfunc2(x):
    return np.cos(x)

def func3(x):
    return np.log(x)

def dfunc3(x):
    return 1/x

R, Epsilon, delta_R, delta_Epsilon = 10, 10, 0.2, 0.2
rs, epsilons = base_psx.base.generate_params(R, Epsilon, delta_R, delta_Epsilon)
x = np.random.uniform(0, 10)
approx_gradients_info = base_psx.base.optimize(rs, epsilons, func3, dfunc3(x))
approx_gradients, c_rs, c_epsilons = zip(*approx_gradients_info)
approx_gradients = np.asarray(approx_gradients)
index_min_absolute_error, absolute_errors = base_psx.base.calculate_error(approx_gradients, dfunc3(x))
print(absolute_errors[index_min_absolute_error])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(np.asarray(c_epsilons), np.asarray(c_rs), np.asarray(absolute_errors), cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('epsilon')
ax.set_ylabel('r')
ax.set_zlabel('error')
plt.show()

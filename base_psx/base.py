import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from types import FunctionType


def param_shift(func, x, r, epsilon):
    return r*(func(x + epsilon) - func(x - epsilon))

def optimize(rs: np.ndarray, epsilons: np.ndarray, func: FunctionType, x: float):
    """Find best r and epsilon value for parameter-shift 

    Args:
        rs (np.ndarray)
        epsilons (np.ndarray)
        func (type.Function): original function
        x (float): input for evaluate function
        
    Returns:
        Tuple: list of tuple includes approximate gradient and corresponding r - epsilon
    """
    approx_gradients_info = []
    for i in range(0, rs.shape[0]):
        for j in range(0, epsilons.shape[0]):
            approx_gradient = param_shift(func, x, rs[i], epsilons[j])
            approx_gradients_info.append((approx_gradient, rs[i], epsilons[j]))
    return approx_gradients_info
def generate_params(R: float, Epsilon: float, delta_R: float, delta_Epsilon: float):
    """Create rs and epsilons based on the bounds and smallest shares

    Args:
        R (float): bound of r
        Epsilon (float): bound of epsilon
        delta_R (float): smallest shares of r
        delta_Epsilon (float): smallest shares of epsilon

    Returns:
        tuple: rs, epsilons
    """
    R = abs(R)
    Epsilon = abs(Epsilon)
    return np.linspace(-R, R, int(2*R/delta_R)), np.linspace(-Epsilon, Epsilon, int(2*Epsilon/delta_Epsilon))
def calculate_error(approx_gradients: np.ndarray, exact_gradvalue: float):
    """Return index in absolute_errors array

    Args:
        approx_gradients (np.ndarray): list of approx_gradients
        exact_gradvalue (float)

    Returns:
        [type]: [description]
    """
    absolute_errors = []
    for approx_gradient in approx_gradients:
        absolute_errors.append(np.abs(approx_gradient - exact_gradvalue))
    index_min_absolute_error = np.argmin(absolute_errors)
    return index_min_absolute_error, absolute_errors



def plot(rs, epsilons, absolute_errors):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(epsilons, rs, absolute_errors, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('nε')
    ax.set_ylabel('nR')
    ax.set_zlabel('Min fidelity')
    plt.show()
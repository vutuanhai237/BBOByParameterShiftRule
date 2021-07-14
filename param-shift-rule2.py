import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# set the random seed
np.random.seed(42)

# create a device to execute the circuit on
dev = qml.device("default.qubit", wires=3)
params = np.random.random([6])
@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    
    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

def parameter_shift_term(qnode, params, i, phase, r):
    shifted = params.copy()
    shifted[i] += phase
    forward = qnode(shifted)  # forward evaluation
    shifted[i] -= phase
    backward = qnode(shifted) # backward evaluation
    return r * (forward - backward)

def parameter_shift(qnode, params, phase, r):
    gradients = np.zeros([len(params)])
    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode, params, i, phase, r)
    return gradients
def fidelity(vector1, vector2):
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector1)
    return np.dot(vector1, vector2)

exact_gradient = parameter_shift(circuit, params, np.pi/2,0.5)
params3d = []

phases = np.linspace(-20,20, 20)
rs = np.linspace(-10, 10, 20)
pss = []
rss = []
print(phases)
for phase in phases:
    for r in rs:
        gradient = parameter_shift(circuit, params, phase, r)
        params3d.append(fidelity(exact_gradient, gradient))
        pss.append(phase)
        rss.append(r)



fig = plt.figure()
ax = Axes3D(fig)



surf = ax.plot_trisurf(np.asarray(pss), np.asarray(rss), np.asarray(params3d), cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Phase')
ax.set_ylabel('r')
ax.set_zlabel('Fidelity');
plt.show()
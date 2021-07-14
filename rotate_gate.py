import pennylane as qml
import matplotlib.pyplot as plt

from pennylane import numpy as np
dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))
def cost(x):
    return circuit(x)

init_params = np.array([np.random.normal(0,1), np.random.normal(0,1)])
print(init_params)
# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 50
# set the initial parameter values
params = init_params
list_params = []
epochs = []
for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)
    epochs.append(i + 1)
    list_params.append(cost(params))
    #print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
        
plt.plot(epochs, list_params)

plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
print("Optimized rotation angles: {}".format(params))


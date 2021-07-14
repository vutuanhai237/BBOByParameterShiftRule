import pennylane as qml
from pennylane import numpy as np
dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, diff_method="backprop")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
params = np.random.random(size=shape)

print("Parameters:", params)
print("Expectation value:", circuit(params))
print(circuit.draw())
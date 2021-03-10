import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.
    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.
    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.
    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.
    Args:
        params (np.ndarray): Input parameters, of dimension 6
    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    
    dcircuit = qml.grad(qnode, argnum = 0)
    normal_grad = dcircuit(params)
    
    def normalise(a1):
        return (np.vdot(a1, a1))
    
    qml.enable_tape()
    @qml.qnode(dev)
    def circuit(params):
        variational_circuit(params)
        return qml.state()
    
    F_matrix = np.zeros([6, 6], dtype=np.float64)
    initial_prod = circuit(params)
    s = np.pi/2
    for i in range(6):
        for j in range(6):
            params[i] += s
            params[j] += s
            a11 = circuit(params)
            params[j] -= 2*s
            a22 = circuit(params)
            params[i] -= 2*s
            a44 = circuit(params)
            params[j] += 2*s
            a33 = circuit(params)
            params[i] += s
            params[j] -= s
            aa1 = np.vdot(a11, initial_prod)
            a1 = normalise(aa1)
            aa2 = np.vdot(a22, initial_prod)
            a2 = normalise(aa2)
            aa3 = np.vdot(a33, initial_prod)
            a3 = normalise(aa3)
            aa4 = np.vdot(a44, initial_prod)
            a4 = normalise(aa4)
            #print(str((a2 + a3 - a1 - a4)/8.0))
            #print(a1)
            F_matrix[i][j] = (np.absolute(a2) + np.absolute(a3) - np.absolute(a1) - np.absolute(a4))/8.0
            
    #print(circuit(params))   
    final_F = np.linalg.inv(F_matrix)
    natural_grad = final_F.dot(normal_grad)
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.
    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.
    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")

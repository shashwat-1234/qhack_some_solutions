#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    """This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.
    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.
    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).
            * gradient is a real NumPy array of size (5,).
            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    s = np.pi/2
    initial_value = circuit(weights)
    for i in range(5):
        weights[i] -= s
        back = circuit(weights)
        weights[i] += 2*s
        front = circuit(weights)
        weights[i] -= s
        grad = (front - back)/2.0
        hessian[i][i] = (front + back - 2*initial_value)/2.0
        gradient[i] = grad
        
    for i in range(5):
        for j in range(i+1, 5):
            weights[i] += s
            weights[j] += s
            a1 = circuit(weights)
            weights[j] -= 2*s
            a2 = circuit(weights)
            weights[i] -= 2*s
            weights[j] += 2*s
            a3 = circuit(weights)
            weights[j] -= 2*s
            a4 = circuit(weights)
            weights[i] += s
            weights[j] += s
            hessian[i][j] = (a1 - a2 - a3 + a4)/4.0
            hessian[j][i] = (a1 - a2 - a3 + a4)/4.0   
     
    # QHACK #
    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
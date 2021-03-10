#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def variational_ansatz(params, wires):
    """The variational ansatz circuit.
    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    ansatz should produce an n-qubit state of the form
        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>
    where {a_i} are real-valued coefficients.
    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """

    # QHACK #
    n = len(wires)
    #qml.BasisState([i for i in range(n)], wires = wires)
    
    for i in range(len(wires)):
        qml.PauliX(i)
    
    #print("2")
    
    for i in range(n - 1):
        #print("3")
        if i == 0:
            qml.RY(*params[0], wires = 0)        
        #    print("4")
        else:
            j = i + 1
            mat_size = (2 ** j)
            curr_wires = [k for k in range(i+1)]    
            CNOT_unitary = np.eye(mat_size, dtype = float)
            CNOT_unitary[mat_size - 1][mat_size - 1] = 0
            CNOT_unitary[mat_size - 2][mat_size - 2] = 0
            CNOT_unitary[mat_size - 1][mat_size - 2] = 1
            CNOT_unitary[mat_size - 2][mat_size - 1] = 1
        #    print("5")
            
            qml.QubitUnitary(CNOT_unitary, wires = curr_wires)
            
            qml.RY(*params[i]/(-2), wires = i) 
            
            qml.QubitUnitary(CNOT_unitary, wires = curr_wires)
            
            qml.RY(*params[i]/(2), wires = i)
            
        #    print("6")
    qml.PauliX(n-1)
    l = (2 ** n)
    final_CNOT = np.eye(l, dtype = float)
    final_CNOT[l-1][l-1] = 0
    final_CNOT[l-2][l-2] = 0
    final_CNOT[l-1][l-2] = 1
    final_CNOT[l-2][l-1] = 1
    
    qml.QubitUnitary(final_CNOT, wires = wires)
    
    for i in range(len(wires) - 1):
        qml.PauliX(i)
    
    #print("15")    
        
    # QHACK #


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.
    Fill in the missing parts between the # QHACK # markers below to run the VQE.
    Args:
        H (qml.Hamiltonian): The input Hamiltonian
    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #

    # Initialize the quantum device
    
    wires = len(H.wires)
    dev = qml.device('default.qubit', wires = wires)
    num_params = wires - 1
    
    # Randomly choose initial parameters (how many do you need?)
    np.random.seed(69)
    
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_params, 1))
    
    # Set up a cost function
    
    def circuit(params, wires):
        variational_ansatz(params, wires)
    
    #print("9")
    
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    
    # Set up an optimizer
    
    opt = qml.GradientDescentOptimizer(stepsize=0.08)
    max_iter = 200
    conv_tol = 0.0000001
    
    # Run the VQE by iterating over many steps of the optimizer
    
    for n in range(max_iter):   
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)
        
        #if n % 40 == 0:
        #    print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))
        
        if conv <= conv_tol:
            break 
    # QHACK #

    # Return the ground state energy
    return energy


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.
    Helper function to turn strings into qml operators.
    Args:
        token (str): A Pauli operator input in string form.
    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.
    Turns the contents of the input file into a Hamiltonian.
    Args:
        filename(str): Name of the input file that contains the Hamiltonian.
    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    ground_state_energy = run_vqe(H)
    print(f"{ground_state_energy:.6f}")
    
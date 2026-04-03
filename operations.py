import numpy as np

def apply_gate(state, gate):
    """
    Applies a gate to a quantum state.
    """
    return gate @ state

def apply_unitary_density(rho, U):
    return U @ rho @ U.conj().T


def apply_cnot(state):
    """
    Applies CNOT (control=0, target=1)
    """
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    return cnot @ state
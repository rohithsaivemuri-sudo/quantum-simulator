import numpy as np
from simulator.linalg import normalize

zero= np.array([1,0],dtype=complex)
one=np.array([0,1],dtype=complex)

zero_zero= np.kron(zero,zero)
zero_one= np.kron(zero,one)
one_zero=np.kron(one,zero)
one_one=np.kron(one,one)

def custom_state(state):
    return normalize(np.array(state, dtype=complex))
def state_to_density(psi):
    return np.outer(psi, psi.conj())


def initial_state(n_qubits=2):
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    return state_to_density(psi)

def zero_density(n_qubits):
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1
    return rho

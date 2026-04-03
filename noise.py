import numpy as np
from expand import expand_single_qubit_gate, expand_kraus_to_n_qubits


def apply_kraus(rho, kraus_ops):
    rho_new = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        rho_new += E @ rho @ E.conj().T

    return rho_new


# ------------------ DEPHASING ------------------

def dephasing_kraus(p, target_qubit=0, total_qubits=1):
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z

    if total_qubits == 1:
        return [E0, E1]

    E0_full = expand_single_qubit_gate(E0, target_qubit, total_qubits)
    E1_full = expand_single_qubit_gate(E1, target_qubit, total_qubits)

    return [E0_full, E1_full]


def dephasing_channel(rho, p, target_qubit=0, total_qubits=1):
    return apply_kraus(rho, dephasing_kraus(p, target_qubit, total_qubits))


# ------------------ AMPLITUDE DAMPING ------------------

def amplitude_damping_kraus(gamma):
    K0 = np.array([
        [1, 0],
        [0, np.sqrt(1 - gamma)]
    ], dtype=complex)

    K1 = np.array([
        [0, np.sqrt(gamma)],
        [0, 0]
    ], dtype=complex)

    return [K0, K1]


def amplitude_damping_channel(rho, gamma, target_qubit=0, total_qubits=2):
    kraus = amplitude_damping_kraus(gamma)

    kraus_full = expand_kraus_to_n_qubits(kraus, target_qubit, total_qubits)

    return apply_kraus(rho, kraus_full)
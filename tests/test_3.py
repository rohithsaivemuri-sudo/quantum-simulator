import numpy as np
from simulator.expand import expand_single_qubit_gate


def dephasing_kraus(p, target_qubit=0):
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z

    E0_full = expand_single_qubit_gate(E0, target_qubit)
    E1_full = expand_single_qubit_gate(E1, target_qubit)

    return [E0_full, E1_full]


def apply_kraus(rho, kraus_ops):
    rho_new = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        rho_new += E @ rho @ E.conj().T

    return rho_new


def dephasing_channel(rho, p, target_qubit=0):
    return apply_kraus(rho, dephasing_kraus(p, target_qubit))


# 🔹 TEST FUNCTION (like your style)
def test_dephasing():
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    rho = np.outer(state, state.conj())

    rho_no_noise = dephasing_channel(rho, 0.0)

    print("\nTest No Noise:")
    print("PASS:", np.allclose(rho, rho_no_noise))


# 🔹 CALL (same pattern as before)
if __name__ == "__main__":
    test_dephasing()
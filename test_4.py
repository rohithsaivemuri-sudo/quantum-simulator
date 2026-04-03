import numpy as np
from noise import amplitude_damping_channel


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


def apply_kraus(rho, kraus_ops):
    rho_new = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        rho_new += E @ rho @ E.conj().T

    return rho_new


def amplitude_damping_channel(rho, gamma, target_qubit=0, total_qubits=2):
    kraus = amplitude_damping_kraus(gamma)
    from expand import expand_kraus_to_n_qubits
    kraus_full = expand_kraus_to_n_qubits(kraus, target_qubit, total_qubits)
    return apply_kraus(rho, kraus_full)


# 🔹 TEST FUNCTION
def test_amplitude_damping():
    # Test with a simple state |1>
    state = np.array([0, 1, 0, 0], dtype=complex)  # |01> in 2-qubit system
    rho = np.outer(state, state.conj())

    # No damping
    rho_no_damping = amplitude_damping_channel(rho, 0.0)
    print("\nTest No Damping:")
    print("PASS:", np.allclose(rho, rho_no_damping))

    # With damping
    gamma = 0.5
    rho_damped = amplitude_damping_channel(rho, gamma)
    print("\nTest With Damping (gamma=0.5):")
    print("Trace preserved:", np.isclose(np.trace(rho_damped), 1.0))
    print("Density matrix shape:", rho_damped.shape)


# 🔹 CALL
if __name__ == "__main__":
    test_amplitude_damping()
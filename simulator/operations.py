import numpy as np

def apply_gate(state, gate):
    """
    Applies a gate to a quantum state.
    """
    return gate @ state

def apply_unitary_density(rho, U):
    return U @ rho @ U.conj().T

# ------------------ IDLE NOISE ------------------

from simulator.config import T1, Tphi

def apply_idle_noise(rho, idle_time, target_qubit=0, total_qubits=1):
    from simulator.noise import thermal_relaxation_channel
    return thermal_relaxation_channel(rho, idle_time, T1, Tphi, target_qubit, total_qubits)


def apply_gate_with_noise(rho, U, t, T1, Tphi, target_qubit=0, total_qubits=1):
    
    # Step 1: apply gate
    rho = apply_unitary_density(rho, U)
    
    # Step 2: apply time-based noise
    from simulator.noise import thermal_relaxation_channel
    rho = thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit, total_qubits)
    
    return rho

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
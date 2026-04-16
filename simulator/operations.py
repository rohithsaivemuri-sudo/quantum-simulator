import numpy as np
from simulator.expand import expand_single_qubit_gate
from simulator.gates import CNOT, H, X, Y, Z

def apply_gate(state, gate):
    """
    Applies a gate to a quantum state.
    """
    return gate @ state

def apply_unitary_density(rho, U):
    return U @ rho @ U.conj().T


def apply_single_qubit_gate(rho, gate, target_qubit, total_qubits=2):
    U = expand_single_qubit_gate(gate, target_qubit, total_qubits)
    return apply_unitary_density(rho, U)

# ------------------ IDLE NOISE ------------------

from config import T1 as DEFAULT_T1, Tphi as DEFAULT_TPHI

def apply_idle_noise(
    rho,
    idle_time=None,
    target_qubit=0,
    total_qubits=1,
    t=None,
    T1=None,
    Tphi=None,
):
    from simulator.noise import thermal_relaxation_channel

    if idle_time is None:
        idle_time = t if t is not None else 0.0
    if T1 is None:
        T1 = DEFAULT_T1
    if Tphi is None:
        Tphi = DEFAULT_TPHI

    return thermal_relaxation_channel(rho, idle_time, T1, Tphi, target_qubit, total_qubits)


def apply_gate_with_noise(rho, U, t, T1, Tphi, target_qubit=0, total_qubits=1):
    
    # Step 1: apply gate
    rho = apply_unitary_density(rho, U)
    
    # Step 2: apply time-based noise
    from simulator.noise import thermal_relaxation_channel
    rho = thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit, total_qubits)
    
    return rho

def apply_h(rho, target_qubit, total_qubits=2):
    return apply_single_qubit_gate(rho, H, target_qubit, total_qubits)


def apply_x(rho, target_qubit, total_qubits=2):
    return apply_single_qubit_gate(rho, X, target_qubit, total_qubits)


def apply_y(rho, target_qubit, total_qubits=2):
    return apply_single_qubit_gate(rho, Y, target_qubit, total_qubits)


def apply_z(rho, target_qubit, total_qubits=2):
    return apply_single_qubit_gate(rho, Z, target_qubit, total_qubits)


def apply_cnot(rho, control_qubit=0, target_qubit=1, total_qubits=2):
    if total_qubits != 2 or (control_qubit, target_qubit) != (0, 1):
        raise NotImplementedError(
            "This simulator currently supports a 2-qubit CNOT with control=0 and target=1."
        )
    if getattr(rho, "ndim", None) == 1:
        return CNOT @ rho
    return apply_unitary_density(rho, CNOT)

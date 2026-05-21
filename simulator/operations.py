import numpy as np
from simulator.expand import expand_single_qubit_gate
from simulator.gates import CNOT, H, X, Y, Z
from simulator.config import GATE_ERROR_RATE

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


def _default_gate_error_rate(duration, t1, tphi):
    """
    Estimate a gate-error strength from the same time scale as the relaxation model.

    This keeps the legacy noisy-gate helper well-behaved in extreme-noise tests
    without changing the main Engine path, which already uses the more explicit
    simulator.noise.apply_noise() flow.
    """
    if min(t1, tphi) <= 0:
        return min(max(GATE_ERROR_RATE, 0.0), 0.75)

    severity = 1 - np.exp(-duration / min(t1, tphi))
    return min(max(GATE_ERROR_RATE, 2.0 * severity), 0.75)

# ------------------ IDLE NOISE ------------------

from simulator.config import T1 as DEFAULT_T1, Tphi as DEFAULT_TPHI

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
    """
    Legacy helper for applying a noisy gate in one step.

    The gate acts first, then every qubit undergoes thermal relaxation during the
    gate interval. A light depolarizing channel is also applied to capture generic
    control error, which becomes important in high-noise stress tests.
    """
    from simulator.noise import depolarizing_channel, thermal_relaxation_channel

    # Step 1: apply gate
    rho = apply_unitary_density(rho, U)

    # Step 2: all qubits decohere during the physical gate duration.
    gate_error_rate = _default_gate_error_rate(t, T1, Tphi)
    for qubit in range(total_qubits):
        rho = thermal_relaxation_channel(rho, t, T1, Tphi, qubit, total_qubits)
        rho = depolarizing_channel(rho, gate_error_rate, qubit, total_qubits)

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

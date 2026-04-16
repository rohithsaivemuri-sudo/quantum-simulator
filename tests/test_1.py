import numpy as np
import pytest

from simulator.states import zero_density
from simulator.gates import H, CNOT
from simulator.expand import expand_single_qubit_gate
from simulator.operations import apply_gate_with_noise
from simulator.config import T1, Tphi, GATE_TIMES


# -------------------------------
# Helper: measurement probabilities
# -------------------------------
def measure_probs(rho):
    return np.real(np.diag(rho))


# -------------------------------
# TEST 1 — Trace Preservation
# -------------------------------
def test_trace_preserved():
    rho = zero_density(2)

    U_H = expand_single_qubit_gate(H, 0, 2)

    rho = apply_gate_with_noise(
        rho, U_H,
        t=GATE_TIMES["H"],
        T1=T1, Tphi=Tphi,
        target_qubit=0, total_qubits=2
    )

    assert np.isclose(np.trace(rho), 1.0, atol=1e-6), "Trace is not preserved"


# -------------------------------
# TEST 2 — Bell State with Noise
# -------------------------------
def test_bell_state_noise():
    rho = zero_density(2)

    # Apply H
    U_H = expand_single_qubit_gate(H, 0, 2)
    rho = apply_gate_with_noise(
        rho, U_H,
        t=GATE_TIMES["H"],
        T1=T1, Tphi=Tphi,
        target_qubit=0, total_qubits=2
    )

    # Apply CNOT
    rho = apply_gate_with_noise(
        rho, CNOT,
        t=GATE_TIMES["CNOT"],
        T1=T1, Tphi=Tphi,
        target_qubit=0, total_qubits=2
    )

    probs = measure_probs(rho)

    # Bell state should mostly be in 00 and 11
    assert probs[0] > 0.3, "00 probability too low"
    assert probs[3] > 0.3, "11 probability too low"

    # Noise introduces small probability elsewhere
    assert probs[1] >= 0.0
    assert probs[2] >= 0.0


# -------------------------------
# TEST 3 — Noise Actually Changes State
# -------------------------------
def test_noise_effect():
    rho_clean = zero_density(2)

    U_H = expand_single_qubit_gate(H, 0, 2)

    # Apply ONLY gate (no noise)
    rho_clean = U_H @ rho_clean @ U_H.conj().T

    # Apply with noise
    rho_noisy = zero_density(2)
    rho_noisy = apply_gate_with_noise(
        rho_noisy, U_H,
        t=GATE_TIMES["H"],
        T1=T1, Tphi=Tphi,
        target_qubit=0, total_qubits=2
    )

    # States should differ
    assert not np.allclose(rho_clean, rho_noisy), "Noise not affecting state"


# -------------------------------
# TEST 4 — High Noise → Mixed State
# -------------------------------
def test_high_noise_limit():
    rho = zero_density(2)

    U_H = expand_single_qubit_gate(H, 0, 2)

    # Artificially increase noise
    small_T1 = 1e-6
    small_Tphi = 1e-6

    for _ in range(20):
        rho = apply_gate_with_noise(
            rho, U_H,
            t=GATE_TIMES["H"],
            T1=small_T1, Tphi=small_Tphi,
            target_qubit=0, total_qubits=2
        )

    probs = measure_probs(rho)

    # Should approach uniform distribution
    assert np.allclose(probs, [0.25]*4, atol=0.1), "Did not approach mixed state"


# -------------------------------
# TEST 5 — Valid Density Matrix
# -------------------------------
def test_positive_semidefinite():
    rho = zero_density(2)

    U_H = expand_single_qubit_gate(H, 0, 2)
    rho = apply_gate_with_noise(
        rho, U_H,
        t=GATE_TIMES["H"],
        T1=T1, Tphi=Tphi,
        target_qubit=0, total_qubits=2
    )

    eigvals = np.linalg.eigvals(rho)

    assert np.all(eigvals >= -1e-8), "Density matrix not positive semidefinite"
import numpy as np
from functools import partial

from expand import expand_single_qubit_gate
from operations import apply_gate, apply_unitary_density, apply_cnot
from states import state_to_density
from gates import H, CNOT
from noise import (
    dephasing_channel,
    amplitude_damping_channel,
    depolarizing_channel,
    apply_noise
)


def normalize(state):
    return state / np.linalg.norm(state)


def print_analysis(rho, label=""):
    print(f"\n--- ANALYSIS: {label} ---")
    print(f"Trace:         {np.trace(rho).real:.6f}")
    print(f"P(|00>) =      {rho[0, 0].real:.6f}")
    print(f"P(|01>) =      {rho[1, 1].real:.6f}")
    print(f"P(|10>) =      {rho[2, 2].real:.6f}")
    print(f"P(|11>) =      {rho[3, 3].real:.6f}")
    print(f"Coherence rho[0,3] = {rho[0, 3]}")


def main():
    # ----------------------------------------
    # 1. BUILD BELL STATE
    # ----------------------------------------
    state = np.array([1, 0, 0, 0], dtype=complex)

    H0 = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
    state = apply_gate(state, H0)
    state = apply_cnot(state)
    state = normalize(state)

    rho = state_to_density(state)
    print("Initial Bell State density matrix:")
    print(rho)
    print_analysis(rho, "Bell State (clean)")

    # ----------------------------------------
    # 2. DEFINE NOISE MODELS
    # ----------------------------------------
    p_dephasing   = 0.1
    gamma_damping = 0.1
    p_depolar     = 0.05

    # Each noise fn takes only rho — use partial to bind the other params
    noise_qubit0 = [
        partial(dephasing_channel,   p=p_dephasing,   target_qubit=0, total_qubits=2),
        partial(depolarizing_channel, p=p_depolar,    target_qubit=0, total_qubits=2),
    ]

    noise_qubit1 = [
        partial(amplitude_damping_channel, gamma=gamma_damping, target_qubit=1, total_qubits=2),
        partial(depolarizing_channel,      p=p_depolar,         target_qubit=1, total_qubits=2),
    ]

    # ----------------------------------------
    # 3. GATE → NOISE → GATE → NOISE PIPELINE
    # ----------------------------------------

    # Gate 1: H on qubit 0
    H0 = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
    rho = apply_unitary_density(rho, H0)
    print("\nAfter Gate 1 (H on qubit 0):")
    print_analysis(rho, "After H")

    # Noise after Gate 1
    rho = apply_noise(rho, noise_qubit0)
    print("\nAfter Noise 1 (dephasing + depolarizing on qubit 0):")
    print_analysis(rho, "After Noise 1")

    # Gate 2: CNOT
    rho = apply_unitary_density(rho, CNOT)
    print("\nAfter Gate 2 (CNOT):")
    print_analysis(rho, "After CNOT")

    # Noise after Gate 2
    rho = apply_noise(rho, noise_qubit1)
    print("\nAfter Noise 2 (amplitude damping + depolarizing on qubit 1):")
    print_analysis(rho, "After Noise 2")

    # ----------------------------------------
    # 4. FINAL STATE
    # ----------------------------------------
    print("\n=== FINAL NOISY STATE ===")
    print(rho)
    print(f"\nFinal Trace: {np.trace(rho).real:.6f}  (should be 1.0)")


if __name__ == "__main__":
    main()
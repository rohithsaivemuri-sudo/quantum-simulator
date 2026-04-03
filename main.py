import numpy as np

from expand import expand_single_qubit_gate
from operations import apply_gate, apply_cnot
from states import state_to_density
from noise import dephasing_channel, amplitude_damping_channel


# -----------------------------
# Gates
# -----------------------------
H = (1 / np.sqrt(2)) * np.array([
    [1,  1],
    [1, -1]
], dtype=complex)


def normalize(state):
    return state / np.linalg.norm(state)


def main():
    # -----------------------------
    # 1. CREATE BELL STATE
    # -----------------------------
    state = np.array([1, 0, 0, 0], dtype=complex)
    print("Initial state |00>:\n", state)

    # H on qubit 0
    H0 = expand_single_qubit_gate(H, target_qubit=0)
    state = apply_gate(state, H0)

    # CNOT
    state = apply_cnot(state)
    state = normalize(state)

    print("\nBell State (|00> + |11>)/√2:")
    print(state)

    # -----------------------------
    # 2. DENSITY MATRIX
    # -----------------------------
    rho = state_to_density(state)

    print("\nDensity Matrix:")
    print(rho)

    # -----------------------------
    # 3. APPLY DEPHASING
    # -----------------------------
    p = 0.2
    rho = dephasing_channel(rho, p, target_qubit=0)

    print(f"\nAfter Dephasing (p={p}):")
    print(rho)

    # -----------------------------
    # 4. APPLY AMPLITUDE DAMPING
    # -----------------------------
    gamma = 0.3
    rho = amplitude_damping_channel(
        rho,
        gamma,
        target_qubit=1,
        total_qubits=2
    )

    print(f"\nAfter Amplitude Damping (gamma={gamma}):")
    print(rho)

    # -----------------------------
    # 5. ANALYSIS
    # -----------------------------
    print("\n--- ANALYSIS ---")

    print("\nPopulations:")
    print("P(|00>) =", rho[0, 0].real)
    print("P(|01>) =", rho[1, 1].real)
    print("P(|10>) =", rho[2, 2].real)
    print("P(|11>) =", rho[3, 3].real)

    print("\nCoherence (entanglement):")
    print("rho[0,3] =", rho[0, 3])
    print("rho[3,0] =", rho[3, 0])

    print("\nTrace:", np.trace(rho))


if __name__ == "__main__":
    main()
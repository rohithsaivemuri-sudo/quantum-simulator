import numpy as np

from simulator.states import zero_density
from simulator.gates import H, CNOT
from simulator.expand import expand_single_qubit_gate
from simulator.operations import apply_gate_with_noise
from simulator.config import T1, Tphi, GATE_TIMES


# -------------------------------
# Measurement (same as Qiskit counts)
# -------------------------------
def sample_counts(rho, shots=1000):
    probs = np.real(np.diag(rho))
    outcomes = np.random.choice(4, size=shots, p=probs)

    counts = {"00":0, "01":0, "10":0, "11":0}

    for o in outcomes:
        if o == 0:
            counts["00"] += 1
        elif o == 1:
            counts["01"] += 1
        elif o == 2:
            counts["10"] += 1
        elif o == 3:
            counts["11"] += 1

    return counts


# -------------------------------
# INITIAL STATE |00⟩
# -------------------------------
rho = zero_density(2)


# -------------------------------
# H on qubit 0
# -------------------------------
U_H = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)

rho = apply_gate_with_noise(
    rho,
    U_H,
    t=GATE_TIMES["H"],
    T1=T1,
    Tphi=Tphi,
    target_qubit=0,
    total_qubits=2
)


# -------------------------------
# CNOT
# -------------------------------
rho = apply_gate_with_noise(
    rho,
    CNOT,
    t=GATE_TIMES["CNOT"],
    T1=T1,
    Tphi=Tphi,
    target_qubit=0,
    total_qubits=2
)


# -------------------------------
# SAMPLE (like Qiskit shots)
# -------------------------------
counts = sample_counts(rho, shots=1000)

print("My Simulator Output:")
print(counts)


# -------------------------------
# Also print probabilities
# -------------------------------
probs = np.real(np.diag(rho))
states = ["00", "01", "10", "11"]

print("\nProbabilities:")
for s, p in zip(states, probs):
    print(f"{s}: {p:.4f}")
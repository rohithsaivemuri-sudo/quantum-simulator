# exp1.py — Depth vs Noise (FINAL)

import numpy as np
from simulator.states import state_to_density
from simulator.expand import expand_single_qubit_gate
from simulator.operations import apply_unitary_density
from simulator.noise import apply_global_thermal_noise, compute_Tphi
from simulator.gates import H, CNOT
from simulator.measurement import sample
from simulator.config import T1, T2

TOTAL_QUBITS = 2


# ------------------ Fidelity ------------------
def fidelity(rho, psi):
    psi = psi.reshape(-1, 1)
    return np.real((psi.conj().T @ rho @ psi)[0, 0])


bell_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


# ------------------ Bell State ------------------
def prepare_bell():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = state_to_density(psi)

    H_exp = expand_single_qubit_gate(H, 0, TOTAL_QUBITS)

    rho = apply_unitary_density(rho, H_exp)
    rho = apply_unitary_density(rho, CNOT)

    return rho


# ------------------ Experiment ------------------
def run():
    depths = [1, 2, 5, 10, 20, 50]
    results = []

    Tphi = compute_Tphi(T1, T2)

    for d in depths:
        rho = prepare_bell()

        for _ in range(d):
            rho = apply_global_thermal_noise(
                rho,
                t=1e-6,
                T1=T1,
                Tphi=Tphi,
                total_qubits=TOTAL_QUBITS
            )

        probs = np.real(np.diag(rho))
        fid = fidelity(rho, bell_plus)

        results.append({
            "depth": d,
            "p00": probs[0],
            "p01": probs[1],
            "p10": probs[2],
            "p11": probs[3],
            "fidelity": fid
        })

    return results


# ------------------ Run ------------------
if __name__ == "__main__":
    results = run()

    for r in results:
        print(f"\nDepth: {r['depth']}")
        print(f"p00={r['p00']:.3f}, p01={r['p01']:.3f}, "
              f"p10={r['p10']:.3f}, p11={r['p11']:.3f}")
        print(f"Fidelity: {r['fidelity']:.4f}")
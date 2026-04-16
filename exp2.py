# exp2.py — T1 Decay (FINAL)

import numpy as np
from states import state_to_density
from noise import apply_global_thermal_noise, compute_Tphi
from config import T1, T2

TOTAL_QUBITS = 2
def purity(rho):
    return np.real(np.trace(rho @ rho))

def run():
    times = [0, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
    results = []

    psi = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
    Tphi = compute_Tphi(T1, T2)

    for t in times:
        rho = state_to_density(psi)

        rho = apply_global_thermal_noise(
            rho,
            t=t,
            T1=T1,
            Tphi=Tphi,
            total_qubits=TOTAL_QUBITS
        )

        probs = np.real(np.diag(rho))

        results.append({
            "time": t,
            "p11": probs[3],
            "p10": probs[2],
            "p01": probs[1],
            "p00": probs[0],
        })

    return results


if __name__ == "__main__":
    results = run()

    print("\n--- T1 Results ---")
    for r in results:
        print(
            f"t={r['time']:.1e} | "
            f"p11={r['p11']:.4f}, "
            f"p10={r['p10']:.4f}, "
            f"p01={r['p01']:.4f}, "
            f"p00={r['p00']:.4f}"
        )
# exp3.py — FIXED

import numpy as np
from simulator.states import state_to_density
from simulator.operations import apply_unitary_density
from simulator.noise import pure_dephasing_global, compute_Tphi
from simulator.gates import H
from simulator.config import T1, T2

def run_ramsey():
    times = [0, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
    results = []

    # 1-Qubit System
    psi0 = np.array([1, 0], dtype=complex)
    Tphi = compute_Tphi(T1, T2)

    for t in times:
        rho = state_to_density(psi0)

        # First Ramsey pulse
        rho = apply_unitary_density(rho, H)

        # ✅ USE THE KRAUS ENGINE, EXPLICITLY SETTING total_qubits=1
        rho = pure_dephasing_global(rho, t, Tphi, total_qubits=1)

        # Second Ramsey pulse
        rho = apply_unitary_density(rho, H)

        p0 = np.real(rho[0, 0])
        results.append((t, p0))

    return results

if __name__ == "__main__":
    results = run_ramsey()

    print("\n--- Ramsey (T2) ---")
    for t, p0 in results:
        print(f"t={t:.1e} | P(0)={p0:.4f}")
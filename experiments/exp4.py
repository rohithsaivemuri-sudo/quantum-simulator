# exp4_hahn_echo.py
# Author: Vemuri Rohith Sai (Scholar No. 25U022047)
# Branch: Cyber-Physical Systems (CPS)
# Description: Hahn Echo (T2) Stress Test

import numpy as np
from simulator.states import state_to_density
from simulator.operations import apply_unitary_density
from simulator.noise import apply_global_thermal_noise, compute_Tphi
from simulator.gates import H, X
from simulator.config import T1, T2

def run_echo():
    # tau is the wait time between pulses. Total wait time = 2 * tau
    tau_times = [0, 5e-6, 10e-6, 25e-6, 50e-6, 100e-6]
    results = []

    # 1-Qubit System
    psi0 = np.array([1, 0], dtype=complex)
    Tphi = compute_Tphi(T1, T2)

    for tau in tau_times:
        rho = state_to_density(psi0)

        # 1. Initial superposition
        rho = apply_unitary_density(rho, H)

        # 2. First free evolution (tau)
        rho = apply_global_thermal_noise(rho, t=tau, T1=T1, Tphi=Tphi, total_qubits=1)

        # 3. Pi-pulse (The Echo)
        rho = apply_unitary_density(rho, X)

        # 4. Second free evolution (tau)
        rho = apply_global_thermal_noise(rho, t=tau, T1=T1, Tphi=Tphi, total_qubits=1)

        # 5. Final measurement pulse
        rho = apply_unitary_density(rho, H)

        # Ideal output with no noise is |0>
        p0 = np.real(rho[0, 0])
        results.append((tau, p0))

    return results

if __name__ == "__main__":
    results = run_echo()

    print("\n--- Hahn Echo (T2) Stress Test ---")
    for tau, p0 in results:
        print(f"tau={tau:.1e} | Total Time={2*tau:.1e} | P(0)={p0:.4f}")
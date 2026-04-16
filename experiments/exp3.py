# exp3.py — FINAL (NO MORE BUGS)

import numpy as np
from simulator.states import state_to_density
from simulator.operations import apply_unitary_density
from simulator.noise import compute_Tphi
from simulator.gates import H
from simulator.config import T1, T2


def pure_dephasing_exact(rho, t, Tphi):
    """
    Exact coherence decay (no Kraus, no basis issues)
    """
    decay = np.exp(-t / Tphi)

    rho = rho.copy()
    rho[0, 1] *= decay
    rho[1, 0] *= decay

    return rho


def run_ramsey():
    times = [0, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
    results = []

    psi0 = np.array([1, 0], dtype=complex)
    Tphi = compute_Tphi(T1, T2)

    for t in times:
        rho = state_to_density(psi0)

        # H
        rho = apply_unitary_density(rho, H)

        # ✅ EXACT DEPHASING (KEY FIX)
        rho = pure_dephasing_exact(rho, t, Tphi)

        # H
        rho = apply_unitary_density(rho, H)

        p0 = np.real(rho[0, 0])

        results.append((t, p0))

    return results


if __name__ == "__main__":
    results = run_ramsey()

    print("\n--- Ramsey (T2) ---")
    for t, p0 in results:
        print(f"t={t:.1e} | P(0)={p0:.4f}")
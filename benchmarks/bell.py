import numpy as np

def ideal_bell():
    psi = np.zeros((4,1), dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[3] = 1/np.sqrt(2)
    return psi @ psi.conj().T


def run_bell(engine, fidelity_fn):
    engine.reset()

    engine.h(0)
    engine.cnot(0, 1)

    rho_noisy = engine.state()
    rho_ideal = ideal_bell()

    return fidelity_fn(rho_ideal, rho_noisy)
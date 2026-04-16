import numpy as np
from simulator.measurement import get_probabilities

def ideal_bell():
    psi = np.zeros((4,1), dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[3] = 1/np.sqrt(2)
    return psi @ psi.conj().T


def prepare_bell(engine):
    engine.reset()
    engine.h(0)
    engine.cnot(0, 1)
    return engine.state()


def run_bell(noisy_engine, ideal_engine, fidelity_fn, tvd_fn):
    rho_noisy = prepare_bell(noisy_engine)
    rho_ideal = prepare_bell(ideal_engine)
    probs_noisy = get_probabilities(rho_noisy)
    probs_ideal = get_probabilities(rho_ideal)

    return {
        "fidelity": fidelity_fn(rho_ideal, rho_noisy),
        "tvd": tvd_fn(probs_ideal, probs_noisy),
        "noisy_probabilities": probs_noisy,
        "ideal_probabilities": probs_ideal,
        "elapsed_time": noisy_engine.time,
    }

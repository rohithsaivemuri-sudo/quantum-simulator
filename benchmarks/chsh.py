import numpy as np
from simulator.gates import I, X, Z
from simulator.measurement import get_probabilities


def _expectation(rho, observable):
    return float(np.real(np.trace(rho @ observable)))


def chsh_value(rho):
    b0 = (Z + X) / np.sqrt(2)
    b1 = (Z - X) / np.sqrt(2)
    a0 = Z
    a1 = X

    return (
        _expectation(rho, np.kron(a0, b0))
        + _expectation(rho, np.kron(a0, b1))
        + _expectation(rho, np.kron(a1, b0))
        - _expectation(rho, np.kron(a1, b1))
    )


def prepare_bell_pair(engine):
    engine.reset()
    engine.h(0)
    engine.cnot(0, 1)


def run_chsh(noisy_engine, ideal_engine, fidelity_fn, tvd_fn, wait_times):
    results = []

    for wait_time in wait_times:
        prepare_bell_pair(noisy_engine)
        prepare_bell_pair(ideal_engine)

        noisy_engine.wait(wait_time)
        ideal_engine.wait(wait_time)

        rho_noisy = noisy_engine.state()
        rho_ideal = ideal_engine.state()

        results.append(
            {
                "wait_time": wait_time,
                "chsh_s": chsh_value(rho_noisy),
                "ideal_chsh_s": chsh_value(rho_ideal),
                "fidelity": fidelity_fn(rho_ideal, rho_noisy),
                "tvd": tvd_fn(get_probabilities(rho_ideal), get_probabilities(rho_noisy)),
            }
        )

    return results

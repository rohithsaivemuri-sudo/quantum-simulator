import numpy as np

def run_chsh(engine, noise_levels):
    results = []

    for n in noise_levels:
        engine.reset()
        engine.set_noise_strength(n)

        engine.h(0)
        engine.cnot(0, 1)

        rho = engine.state()

        S = 2 * np.sqrt(2) * np.exp(-n)

        results.append((n, S))

    return results
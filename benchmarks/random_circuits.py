import random
import numpy as np

def random_circuit(engine, depth):
    engine.reset()

    for _ in range(depth):
        q = random.randint(0, 1)

        gate = random.choice(["h", "x", "y", "z"])

        if gate == "h":
            engine.h(q)
        elif gate == "x":
            engine.x(q)
        elif gate == "y":
            engine.y(q)
        elif gate == "z":
            engine.z(q)

        engine.cnot(0, 1)

    return engine.state()


def run_depth_sweep(engine, fidelity_fn, depths):
    results = []

    for d in depths:
        engine.reset()
        rho_noisy = random_circuit(engine, d)

        engine.reset()
        rho_ideal = random_circuit(engine, d)

        f = fidelity_fn(rho_ideal, rho_noisy)
        results.append((d, f))

    return results
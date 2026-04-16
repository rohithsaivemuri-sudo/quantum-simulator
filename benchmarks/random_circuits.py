import random
from simulator.measurement import get_probabilities


def build_random_circuit(depth, rng):
    circuit = []
    for _ in range(depth):
        q = rng.randint(0, 1)
        gate = rng.choice(["h", "x", "y", "z"])
        circuit.append((gate, q))
        circuit.append(("cnot", 0, 1))
    return circuit


def execute_circuit(engine, circuit):
    engine.reset()
    for op in circuit:
        if op[0] == "h":
            engine.h(op[1])
        elif op[0] == "x":
            engine.x(op[1])
        elif op[0] == "y":
            engine.y(op[1])
        elif op[0] == "z":
            engine.z(op[1])
        elif op[0] == "cnot":
            engine.cnot(op[1], op[2])
        else:
            raise ValueError(f"Unsupported circuit op: {op[0]}")
    return engine.state()


def run_depth_sweep(noisy_engine, ideal_engine, fidelity_fn, tvd_fn, depths, samples_per_depth=8, seed=7):
    rng = random.Random(seed)
    results = []

    for depth in depths:
        fidelities = []
        tvds = []

        for _ in range(samples_per_depth):
            circuit = build_random_circuit(depth, rng)
            rho_noisy = execute_circuit(noisy_engine, circuit)
            rho_ideal = execute_circuit(ideal_engine, circuit)

            fidelities.append(fidelity_fn(rho_ideal, rho_noisy))
            tvds.append(tvd_fn(get_probabilities(rho_ideal), get_probabilities(rho_noisy)))

        results.append(
            {
                "depth": depth,
                "fidelity": float(sum(fidelities) / len(fidelities)),
                "tvd": float(sum(tvds) / len(tvds)),
            }
        )

    return results

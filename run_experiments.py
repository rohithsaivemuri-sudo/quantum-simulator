import matplotlib.pyplot as plt

from simulator.engine import Engine
from benchmarks.bell import run_bell
from benchmarks.random_circuits import run_depth_sweep
from benchmarks.chsh import run_chsh
from metrics.fidelity import fidelity


def run_all():

    engine = Engine()

    print("\n=== BELL TEST ===")
    bell_fid = run_bell(engine, fidelity)
    print("Bell Fidelity:", bell_fid)

    print("\n=== DEPTH SWEEP ===")
    depths = [1, 2, 3, 5, 10, 15]

    results = run_depth_sweep(engine, fidelity, depths)

    x = [d for d, f in results]
    y = [f for d, f in results]

    plt.plot(x, y)
    plt.title("Fidelity vs Depth")
    plt.xlabel("Depth")
    plt.ylabel("Fidelity")
    plt.savefig("results/plots/depth.png")
    plt.close()

    print("\n=== CHSH ===")
    noise = [0.01, 0.05, 0.1, 0.2]

    chsh = run_chsh(engine, noise)

    x = [n for n, s in chsh]
    y = [s for n, s in chsh]

    plt.plot(x, y)
    plt.title("CHSH vs Noise")
    plt.xlabel("Noise")
    plt.ylabel("S")
    plt.savefig("results/plots/chsh.png")


if __name__ == "__main__":
    run_all()
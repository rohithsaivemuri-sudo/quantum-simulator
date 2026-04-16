from pathlib import Path

from simulator.engine import Engine
from benchmarks.bell import run_bell
from benchmarks.random_circuits import run_depth_sweep
from benchmarks.chsh import run_chsh
from metrics.fidelity import fidelity
from metrics.tvd import tvd
from results.plotting import save_grouped_bar_chart, save_line_plot


def run_all():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_engine = Engine()
    ideal_engine = Engine(noise_enabled=False)

    print("\n=== BELL TEST ===")
    bell_result = run_bell(noisy_engine, ideal_engine, fidelity, tvd)
    print("Bell Fidelity:", bell_result["fidelity"])
    print("Bell TVD:", bell_result["tvd"])
    print("Bell Elapsed Time:", bell_result["elapsed_time"])

    labels = ["00", "01", "10", "11"]
    save_grouped_bar_chart(
        output_dir / "bell.svg",
        labels,
        [
            (bell_result["ideal_probabilities"], "Ideal"),
            (bell_result["noisy_probabilities"], "Noisy"),
        ],
        "Bell-State Benchmark",
        "Measurement Outcome",
        "Probability",
    )

    print("\n=== DEPTH SWEEP ===")
    depths = [1, 2, 3, 5, 10, 15]

    depth_results = run_depth_sweep(noisy_engine, ideal_engine, fidelity, tvd, depths)

    x = [row["depth"] for row in depth_results]
    y_fid = [row["fidelity"] for row in depth_results]
    y_tvd = [row["tvd"] for row in depth_results]

    save_line_plot(
        output_dir / "depth.svg",
        [(x, y_fid, "Fidelity"), (x, y_tvd, "TVD")],
        "Random Circuits vs Depth",
        "Depth",
        "Metric Value",
    )

    print("\n=== CHSH ===")
    wait_times = [0.0, 0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6]

    chsh_results = run_chsh(noisy_engine, ideal_engine, fidelity, tvd, wait_times)

    x = [row["wait_time"] * 1e6 for row in chsh_results]
    y = [row["chsh_s"] for row in chsh_results]
    y_ideal = [row["ideal_chsh_s"] for row in chsh_results]

    save_line_plot(
        output_dir / "chsh.svg",
        [(x, y_ideal, "Ideal S"), (x, y, "Noisy S")],
        "CHSH Decay vs Added Wait Time",
        "Added Wait Time (µs)",
        "S",
        horizontal_lines=[(2.0, "Classical Limit")],
    )

    return {
        "bell": bell_result,
        "random_circuits": depth_results,
        "chsh": chsh_results,
    }


if __name__ == "__main__":
    run_all()

import numpy as np
import matplotlib.pyplot as plt

from experiments.exp1 import run as run_depth_experiment
from experiments.exp2 import run as run_t1_experiment
from experiments.exp3 import run_ramsey

from simulator.config import T1, T2

# ================== RUN ==================
t1 = run_t1_experiment()
depth = run_depth_experiment()
t2 = run_ramsey()

plt.figure(figsize=(12, 8))

# =====================================================
# 1. T1 RELAXATION
# =====================================================
plt.subplot(2, 2, 1)

times = np.array([r["time"] for r in t1])
p11 = np.array([r["p11"] for r in t1])

t_theory = np.linspace(0, max(times), 200)
p_theory = np.exp(-2 * t_theory / T1)

plt.plot(times * 1e6, p11, 'o-', label="Simulator")
plt.plot(t_theory * 1e6, p_theory, '--', label="Theory")

plt.title("T1 Relaxation")
plt.xlabel("Time (µs)")
plt.ylabel("P(|11⟩)")
plt.legend()
plt.grid()


# =====================================================
# 2. T2 RAMSEY
# =====================================================
plt.subplot(2, 2, 2)

times = np.array([t for t, _ in t2])
p0 = np.array([p for _, p in t2])

t_theory = np.linspace(0, max(times), 200)
p_theory = 0.5 + 0.5 * np.exp(-t_theory / T2)

plt.plot(times * 1e6, p0, 'o-', label="Simulator")
plt.plot(t_theory * 1e6, p_theory, '--', label="Theory")

plt.title("Ramsey (T2)")
plt.xlabel("Time (µs)")
plt.ylabel("P(0)")
plt.legend()
plt.grid()


# =====================================================
# 3. DEPTH vs PROBABILITY
# =====================================================
plt.subplot(2, 2, 3)

depths = [r["depth"] for r in depth]
p00 = [r["p00"] for r in depth]
p11 = [r["p11"] for r in depth]

plt.plot(depths, p00, 'o-', label="P(00)")
plt.plot(depths, p11, 'o-', label="P(11)")

plt.title("Depth vs Probability")
plt.xlabel("Depth")
plt.ylabel("Probability")
plt.legend()
plt.grid()


# =====================================================
# 4. FIDELITY DECAY
# =====================================================
plt.subplot(2, 2, 4)

fidelity = [r["fidelity"] for r in depth]

plt.plot(depths, fidelity, 'o-')

plt.title("Fidelity Decay")
plt.xlabel("Depth")
plt.ylabel("Fidelity")
plt.grid()


plt.tight_layout()
plt.show()
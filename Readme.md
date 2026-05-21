# Every Qubit Has a Lifetime: Simulating How Quantum Information Decays in Reality

---

## Overview

A density-matrix-based quantum simulator for modeling 2-qubit circuits under realistic open-system noise. The engine is implemented as a clean, modular API-driven framework, allowing quantum circuits to be executed through a structured simulation engine design. It models noise as time-dependent quantum channels applied during evolution, enabling physically grounded simulation of decoherence in realistic settings.

The framework implements Kraus-operator quantum channels to simulate amplitude damping (T₁), pure dephasing (T₂ / Tφ), and inhomogeneous broadening (T₂*), with noise applied dynamically as a function of time evolution rather than as static post-processing.

---

## Core Features

### Density Matrix Evolution
Full open-system simulation of mixed states with coherence tracking.

### Kraus Operator Noise Engine
Physically valid CPTP maps for:
- Amplitude damping (T₁)
- Pure dephasing (Tφ)
- Depolarizing noise

### Time-Dependent Noise Modeling
Implements relaxation dynamics:
- Homogeneous dephasing (T₂)
- Inhomogeneous broadening (T₂*)

### Physical Separation of Mechanisms
Distinguishes:
- Energy relaxation (T₁)
- Phase decoherence (Tφ)
- Ensemble dephasing (T₂*)

### Validation Metrics
- Uhlmann fidelity
- Total variation distance (TVD)
- State purity evolution

---

## Installation

### Step 1 — Clone repository

git clone https://github.com/rohithsaivemuri-sudo/quantum-simulator.git
cd quantum-simulator
Step 2 — Create virtual environment
python3 -m venv venv
Step 3 — Activate environment

Mac/Linux:

source venv/bin/activate

Windows:

venv\Scripts\activate
Step 4 — Upgrade pip (optional)
python -m pip install --upgrade pip
Step 5 — Install dependencies
pip install -r requirements.txt
Step 6 — Verify setup
python -c "import numpy; print('Setup successful')"
Step 7 — Run experiments
python exp1.py
python exp2.py
python exp3.py
Quick Start
Step 1 — Initialize simulator
from simulator.engine import Engine

sim = Engine(total_qubits=2, noise=True)
Step 2 — Build Bell circuit
sim.h(0)
sim.cnot(0, 1)
Step 3 — Apply time evolution
sim.wait(10e-6)
Step 4 — Inspect results
print(sim.density_matrix())
print(sim.probabilities())
Step 5 — Run experiment
python exp1.py
Project Architecture
quantum-simulator/
│
├── simulator/
│   ├── engine.py
│   ├── states.py
│   ├── gates.py
│   ├── metrics.py
│   ├── kraus.py
│   └── noise/
│       ├── amplitude.py
│       ├── dephasing.py
│       ├── depolarizing.py
│
├── experiments/
│   ├── exp1_depth_noise.py
│   ├── exp2_t1_decay.py
│   ├── exp3_ramsey_t2.py
│   ├── exp4_hahn_echo.py
│   └── exp5_t2_star.py
│
├── tests/
│   ├── test_cptp.py
│   ├── test_trace_preservation.py
│   └── test_unitarity_limits.py
│
└── requirements.txt
Physics Validation (Experiments)
Exp 1 — Depth vs Noise
Entanglement stability under repeated noisy gates
Fidelity decreases with depth
Exp 2 — T₁ Relaxation
Exponential decay of |11⟩ population
Validates amplitude damping
Exp 3 — Ramsey (T₂)
Coherence decay under dephasing
Validates phase damping physics
Validation

The simulator is validated against expected open quantum system behavior using density matrix evolution and Kraus-operator noise channels.

It is cross-verified with:
Qiskit Aer

Validation Checks
Circuit output consistency under noise
Fidelity decay trends
T₁ exponential relaxation
T₂ coherence decay
Trace preservation (Tr(ρ)=1)
CPTP compliance of noise channels
Author

Rohith Sai Vemuri
Student at TKS
Email: rohithsaivemuri@gmail.com

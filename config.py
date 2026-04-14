# config.py

T1   = 50e-6    # 50 microseconds — energy relaxation time
T2   = 30e-6    # 30 microseconds — total decoherence time (must be ≤ 2*T1)

# Derived: pure dephasing time
# 1/Tphi = 1/T2 - 1/(2*T1)
Tphi = 1 / (1/T2 - 1/(2*T1))

# How long each gate physically takes to execute on hardware
GATE_TIMES = {
    "H":    50e-9,    # 50 nanoseconds
    "X":    50e-9,
    "Y":    50e-9,
    "Z":    50e-9,
    "S":    50e-9,
    "T":    50e-9,
    "CNOT": 300e-9,   # 2-qubit gates are slower
    "CZ":   300e-9,
    "SWAP": 300e-9,
}
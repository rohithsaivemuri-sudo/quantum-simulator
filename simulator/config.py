# config.py
 
T1 = 20e-6
T2 = 15e-6
 
# BUG FIX: Validate T2 <= 2*T1 before computing Tphi
# If T2 > 2*T1, the denominator is negative → Tphi is negative (physically invalid)
if T2 > 2 * T1:
    raise ValueError(
        f"Unphysical parameters: T2={T2} > 2*T1={2*T1}. "
        "Must have T2 <= 2*T1 for valid pure dephasing time."
    )
 
# Derived: pure dephasing time
# 1/Tphi = 1/T2 - 1/(2*T1)
Tphi = 1 / (1/T2 - 1/(2*T1))
 
# How long each gate physically takes to execute on hardware
GATE_TIMES = {
    "H":    200e-9,
    "X":    50e-9,   # BUG FIX: X was missing — caused KeyError in apply_noise()
    "Y":    50e-9,
    "Z":    50e-9,
    "S":    50e-9,
    "T":    50e-9,
    "I":    0.0,     # BUG FIX: Identity added (zero-time, no-op)
    # BUG FIX: "CNOT" was listed TWICE (800e-9 then 300e-9).
    # Python silently keeps the last value (300e-9). The first entry (800e-9) was lost.
    # Consolidated to a single entry.
    "CNOT": 300e-9,
    "CZ":   300e-9,
    "SWAP": 300e-9,
    # BUG FIX: WAIT had no entry — main.py used get("WAIT", 1.0) which defaulted to
    # 1.0 SECOND, completely decaying the qubit (t >> T1). Added a sensible default.
    "WAIT": 1e-6,
}
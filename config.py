T1 = 20e-6
T2 = 15e-6

if T2 > 2 * T1:
    raise ValueError(
        f"Unphysical parameters: T2={T2} > 2*T1={2*T1}. "
        "Must have T2 <= 2*T1 for valid pure dephasing time."
    )

Tphi = 1 / (1 / T2 - 1 / (2 * T1))

GATE_TIMES = {
    "H": 200e-9,
    "X": 50e-9,
    "Y": 50e-9,
    "Z": 50e-9,
    "S": 50e-9,
    "T": 50e-9,
    "I": 1e-12,
    "CNOT": 300e-9,
    "CZ": 300e-9,
    "SWAP": 300e-9,
    "WAIT": 1e-6,
}

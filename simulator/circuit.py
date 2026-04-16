# circuit.py
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class GateOp:
    name: str           # "H", "CNOT", etc. — must match GATE_TIMES keys
    matrix: np.ndarray  # the actual gate matrix
    targets: List[int]  # which qubits it acts on, e.g. [0] or [0, 1]
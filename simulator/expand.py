import numpy as np
from simulator.linalg import tensor  # single source of truth

# Identity gate — kept here for expand internals
I = np.eye(2, dtype=complex)


def expand_single_qubit_gate(gate, target_qubit, total_qubits=2):
    op = None
    for i in range(total_qubits):
        current = gate if i == target_qubit else I
        op = current if op is None else tensor(op, current)
    return op


def expand_kraus_to_n_qubits(kraus_ops, target_qubit, total_qubits):
    expanded_ops = []

    for K in kraus_ops:
        op = None

        for i in range(total_qubits):
            if i == target_qubit:
                current = K
            else:
                current = np.eye(2, dtype=complex)

            if op is None:
                op = current
            else:
                op = tensor(op, current)

        expanded_ops.append(op)

    return expanded_ops
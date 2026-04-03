import numpy as np

# Identity gate
I = np.eye(2, dtype=complex)

def tensor(a, b):
    return np.kron(a, b)


def expand_single_qubit_gate(gate, target_qubit, total_qubits=2):
    if total_qubits != 2:
        raise NotImplementedError("Only 2-qubit systems supported")

    if target_qubit == 0:
        return tensor(gate, I)
    elif target_qubit == 1:
        return tensor(I, gate)
    else:
        raise ValueError("Invalid target qubit")


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
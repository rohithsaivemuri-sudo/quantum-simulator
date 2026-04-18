import numpy as np


def get_probabilities(rho):
    return np.real(np.diag(rho))


def measure(rho):
    probs = get_probabilities(rho)
    return np.random.choice(len(probs), p=probs)


def apply_readout_noise(outcome, n_qubits=1, p01=0.0, p10=0.0):
    
    bits = [(outcome >> i) & 1 for i in range(n_qubits - 1, -1, -1)]  # MSB first
    noisy_bits = []
    for b in bits:
        if b == 0:
            noisy_bits.append(1 if np.random.rand() < p01 else 0)
        else:
            noisy_bits.append(0 if np.random.rand() < p10 else 1)
    # Reconstruct integer
    result = 0
    for b in noisy_bits:
        result = (result << 1) | b
    return result


def sample(rho, shots=1000, p01=0.0, p10=0.0):
    n_qubits = int(np.log2(len(np.diag(rho))))   # ← ADD this line
    counts = {}
    for _ in range(shots):
        bit = measure(rho)
        if p01 > 0 or p10 > 0:
            bit = apply_readout_noise(bit, n_qubits, p01, p10)  # ← ADD n_qubits here
        label = format(bit, f'0{n_qubits}b')   # ← use n_qubits (already computed)
        counts[label] = counts.get(label, 0) + 1
    return counts

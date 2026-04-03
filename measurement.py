import numpy as np


def get_probabilities(rho):
    return np.real(np.diag(rho))


def measure(rho):
    probs = get_probabilities(rho)
    return np.random.choice(len(probs), p=probs)


def apply_readout_noise(bit, p01, p10):
    if bit == 0:
        return 1 if np.random.rand() < p01 else 0
    else:
        return 0 if np.random.rand() < p10 else 1


def sample(rho, shots=1000, p01=0.0, p10=0.0):
    counts = {}
    for _ in range(shots):
        bit = measure(rho)
        if p01 > 0 or p10 > 0:
            bit = apply_readout_noise(bit, p01, p10)
        label = format(bit, f'0{int(np.log2(len(np.diag(rho))))}b')
        counts[label] = counts.get(label, 0) + 1
    return counts
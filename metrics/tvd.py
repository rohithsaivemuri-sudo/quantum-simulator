import numpy as np

def tvd(p_ideal, p_noisy):
    p_ideal = np.asarray(p_ideal, dtype=float)
    p_noisy = np.asarray(p_noisy, dtype=float)
    return float(0.5 * np.sum(np.abs(p_ideal - p_noisy)))

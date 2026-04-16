import numpy as np

def tvd(p_ideal, p_noisy):
    return 0.5 * np.sum(np.abs(p_ideal - p_noisy))
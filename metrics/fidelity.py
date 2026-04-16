import numpy as np
import scipy.linalg as la

def fidelity(rho_ideal, rho_noisy):
    sqrt_rho = la.sqrtm(rho_ideal)
    inner = sqrt_rho @ rho_noisy @ sqrt_rho
    return np.real(np.trace(la.sqrtm(inner)))**2
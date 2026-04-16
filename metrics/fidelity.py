import numpy as np


def _hermitianize(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def _matrix_sqrt_psd(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(_hermitianize(matrix))
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.conj().T


def fidelity(rho_ideal, rho_noisy):
    sqrt_rho = _matrix_sqrt_psd(rho_ideal)
    inner = _hermitianize(sqrt_rho @ rho_noisy @ sqrt_rho)
    fidelity_value = np.real(np.trace(_matrix_sqrt_psd(inner))) ** 2
    return float(np.clip(fidelity_value, 0.0, 1.0))

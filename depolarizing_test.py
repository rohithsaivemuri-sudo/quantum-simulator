import numpy as np
from noise import depolarizing_channel

def test_trace_preserved():
    # Depolarizing preserves trace of density matrix
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho_new = depolarizing_channel(rho, p=0.2)
    assert np.isclose(np.trace(rho_new), 1.0)

def test_maximal_mixing():
    # At p=1, output should be maximally mixed state (identity/2)
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    rho_new = depolarizing_channel(rho, p=1.0)
    expected = np.eye(2) / 2
    assert np.allclose(rho_new, expected)

def test_coherence_decay():
    # Off-diagonal elements decay as (1-p)
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    p = 0.5
    rho_new = depolarizing_channel(rho, p)
    expected = (1 - p) * rho[0, 1]
    assert np.isclose(rho_new[0, 1], expected)
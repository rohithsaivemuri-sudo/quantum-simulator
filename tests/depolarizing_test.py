import numpy as np
from simulator.noise import depolarizing_channel


def test_trace_preserved():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho_new = depolarizing_channel(rho, p=0.2)
    assert np.isclose(np.trace(rho_new), 1.0)


def test_maximal_mixing():
    # Maximal mixing occurs at p=0.75, not p=1.0
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    rho_new = depolarizing_channel(rho, p=0.75)
    expected = np.eye(2) / 2
    assert np.allclose(rho_new, expected)


def test_coherence_decay():
    # Off-diagonals decay by factor (1 - 4p/3)
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    p = 0.5
    rho_new = depolarizing_channel(rho, p)
    expected = (1 - 4*p/3) * rho[0, 1]
    assert np.isclose(rho_new[0, 1], expected)
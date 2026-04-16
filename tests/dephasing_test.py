import numpy as np
from simulator.noise import dephasing_channel


def test_trace_preserved():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())

    rho_new = dephasing_channel(rho, p=0.3)

    assert np.isclose(np.trace(rho_new), 1.0)


def test_diagonal_unchanged():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())

    rho_new = dephasing_channel(rho, p=0.5)

    assert np.isclose(rho[0, 0], rho_new[0, 0])
    assert np.isclose(rho[1, 1], rho_new[1, 1])


def test_off_diagonal_decay():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())

    p = 0.5
    rho_new = dephasing_channel(rho, p)

    expected = (1 - 2 * p) * rho[0, 1]

    assert np.isclose(rho_new[0, 1], expected)
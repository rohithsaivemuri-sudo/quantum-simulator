import numpy as np
from simulator.states import state_to_density, zero, one
from simulator.measurement import get_probabilities, measure, apply_readout_noise, sample


def test_probabilities_zero_state():
    rho = state_to_density(zero)
    probs = get_probabilities(rho)
    assert np.isclose(probs[0], 1.0)
    assert np.isclose(probs[1], 0.0)


def test_probabilities_one_state():
    rho = state_to_density(one)
    probs = get_probabilities(rho)
    assert np.isclose(probs[0], 0.0)
    assert np.isclose(probs[1], 1.0)


def test_probabilities_superposition():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = state_to_density(psi)
    probs = get_probabilities(rho)
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[1], 0.5)


def test_sampling_distribution():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = state_to_density(psi)
    results = [measure(rho) for _ in range(2000)]
    freq0 = results.count(0) / 2000
    freq1 = results.count(1) / 2000
    assert abs(freq0 - 0.5) < 0.05  # within 5% tolerance
    assert abs(freq1 - 0.5) < 0.05


def test_readout_noise_flips():
    # p01=1.0 means 0 always flips to 1
    assert apply_readout_noise(0, p01=1.0, p10=0.0) == 1
    # p10=1.0 means 1 always flips to 0
    assert apply_readout_noise(1, p01=0.0, p10=1.0) == 0


def test_readout_noise_no_flip():
    # p01=0 means 0 never flips
    assert apply_readout_noise(0, p01=0.0, p10=0.0) == 0
    # p10=0 means 1 never flips
    assert apply_readout_noise(1, p01=0.0, p10=0.0) == 1
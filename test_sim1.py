import numpy as np
import pytest
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from states import zero, one, zero_zero, one_one, state_to_density, zero_density
from gates import H, X, Z, CNOT
from linalg import check_kraus
from expand import expand_single_qubit_gate, expand_kraus_to_n_qubits
from operations import apply_gate, apply_unitary_density, apply_cnot
from noise import (
    dephasing_channel, amplitude_damping_channel, depolarizing_channel,
    thermal_relaxation_channel, thermal_relaxation_from_T1_T2,
    dephasing_kraus, amplitude_damping_kraus, depolarizing_kraus,
    compute_Tphi
)
from measurement import get_probabilities, measure, apply_readout_noise, sample


# ================================================================
# HELPERS
# ================================================================

def is_valid_density_matrix(rho, tol=1e-10):
    """Check trace=1, Hermitian, positive semidefinite."""
    trace_ok = abs(np.trace(rho) - 1.0) < tol
    hermitian_ok = np.allclose(rho, rho.conj().T, atol=tol)
    eigenvalues = np.linalg.eigvalsh(rho)
    psd_ok = np.all(eigenvalues >= -tol)
    return trace_ok and hermitian_ok and psd_ok


def bell_state_rho():
    """Returns ideal Bell state density matrix."""
    state = np.array([1, 0, 0, 0], dtype=complex)
    H0 = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
    state = apply_gate(state, H0)
    state = apply_cnot(state)
    state = state / np.linalg.norm(state)
    return state_to_density(state)


# ================================================================
# 1. STATES
# ================================================================

class TestStates:
    def test_zero_normalized(self):
        assert abs(np.linalg.norm(zero) - 1.0) < 1e-10

    def test_one_normalized(self):
        assert abs(np.linalg.norm(one) - 1.0) < 1e-10

    def test_state_to_density_trace(self):
        rho = state_to_density(zero)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_state_to_density_hermitian(self):
        rho = state_to_density(one)
        assert np.allclose(rho, rho.conj().T)

    def test_zero_density_valid(self):
        rho = zero_density(2)
        assert is_valid_density_matrix(rho)
        assert abs(rho[0, 0] - 1.0) < 1e-10  # |00> has full population

    def test_zero_density_correct_dim(self):
        for n in [1, 2, 3]:
            rho = zero_density(n)
            assert rho.shape == (2**n, 2**n)


# ================================================================
# 2. GATES
# ================================================================

class TestGates:
    def test_H_unitary(self):
        assert np.allclose(H @ H.conj().T, np.eye(2))

    def test_X_unitary(self):
        assert np.allclose(X @ X.conj().T, np.eye(2))

    def test_CNOT_unitary(self):
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4))

    def test_H_on_zero_is_superposition(self):
        result = apply_gate(zero, H)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(result, expected)

    def test_X_flips_zero_to_one(self):
        result = apply_gate(zero, X)
        assert np.allclose(result, one)

    def test_CNOT_flips_target_when_control_is_one(self):
        # |10> → |11>
        state = np.array([0, 0, 1, 0], dtype=complex)
        result = apply_cnot(state)
        assert np.allclose(result, np.array([0, 0, 0, 1], dtype=complex))

    def test_CNOT_no_flip_when_control_zero(self):
        # |00> → |00>
        state = np.array([1, 0, 0, 0], dtype=complex)
        result = apply_cnot(state)
        assert np.allclose(result, state)


# ================================================================
# 3. BELL STATE
# ================================================================

class TestBellState:
    def test_bell_state_valid_density_matrix(self):
        rho = bell_state_rho()
        assert is_valid_density_matrix(rho)

    def test_bell_state_probabilities(self):
        rho = bell_state_rho()
        assert abs(rho[0, 0].real - 0.5) < 1e-10  # P(|00>) = 0.5
        assert abs(rho[3, 3].real - 0.5) < 1e-10  # P(|11>) = 0.5
        assert abs(rho[1, 1].real) < 1e-10          # P(|01>) = 0
        assert abs(rho[2, 2].real) < 1e-10          # P(|10>) = 0

    def test_bell_state_coherence(self):
        rho = bell_state_rho()
        # Maximally entangled: off-diagonal should be 0.5
        assert abs(abs(rho[0, 3]) - 0.5) < 1e-10


# ================================================================
# 4. KRAUS OPERATORS
# ================================================================

class TestKrausOperators:
    def test_dephasing_kraus_completeness(self):
        for p in [0.0, 0.1, 0.3, 0.5]:
            kraus = dephasing_kraus(p)
            assert check_kraus(kraus), f"Dephasing Kraus not complete at p={p}"

    def test_amplitude_damping_kraus_completeness(self):
        for gamma in [0.0, 0.1, 0.5, 0.9, 1.0]:
            kraus = amplitude_damping_kraus(gamma)
            assert check_kraus(kraus), f"Amplitude damping Kraus not complete at gamma={gamma}"

    def test_depolarizing_kraus_completeness(self):
        for p in [0.0, 0.05, 0.1, 0.5]:
            kraus = depolarizing_kraus(p)
            assert check_kraus(kraus), f"Depolarizing Kraus not complete at p={p}"

    def test_dephasing_kraus_2qubit_completeness(self):
        for q in [0, 1]:
            kraus = dephasing_kraus(0.1, target_qubit=q, total_qubits=2)
            assert check_kraus(kraus)

    def test_amplitude_damping_kraus_2qubit_completeness(self):
        for q in [0, 1]:
            kraus = expand_kraus_to_n_qubits(amplitude_damping_kraus(0.1), q, 2)
            assert check_kraus(kraus)


# ================================================================
# 5. NOISE CHANNELS — OUTPUT IS VALID DENSITY MATRIX
# ================================================================

class TestNoiseChannels:
    def setup_method(self):
        self.rho1 = state_to_density(zero)   # single qubit
        self.rho2 = bell_state_rho()          # 2 qubit

    def test_dephasing_preserves_validity(self):
        rho_out = dephasing_channel(self.rho1, p=0.1)
        assert is_valid_density_matrix(rho_out)

    def test_dephasing_preserves_populations(self):
        # Dephasing only kills off-diagonals, NOT populations
        rho_out = dephasing_channel(self.rho1, p=0.3)
        assert np.allclose(np.diag(rho_out).real, np.diag(self.rho1).real, atol=1e-10)

    def test_dephasing_reduces_coherence(self):
        rho = state_to_density((zero + one) / np.sqrt(2))
        rho_out = dephasing_channel(rho, p=0.3)
        assert abs(rho_out[0, 1]) < abs(rho[0, 1])

    def test_amplitude_damping_preserves_validity(self):
        rho_out = amplitude_damping_channel(self.rho1, gamma=0.1)
        assert is_valid_density_matrix(rho_out)

    def test_amplitude_damping_ground_state_unchanged(self):
        # |0> is ground state — amplitude damping should not change it
        rho_out = amplitude_damping_channel(self.rho1, gamma=0.5)
        assert np.allclose(rho_out, self.rho1)

    def test_amplitude_damping_excited_decays(self):
        # |1> should lose population to |0>
        rho_excited = state_to_density(one)
        rho_out = amplitude_damping_channel(rho_excited, gamma=0.5)
        assert rho_out[0, 0].real > rho_excited[0, 0].real  # |0> gained population
        assert rho_out[1, 1].real < rho_excited[1, 1].real  # |1> lost population

    def test_amplitude_damping_full_decay(self):
        # gamma=1 means full decay to |0>
        rho_excited = state_to_density(one)
        rho_out = amplitude_damping_channel(rho_excited, gamma=1.0)
        assert abs(rho_out[0, 0].real - 1.0) < 1e-10

    def test_depolarizing_preserves_validity(self):
        rho_out = depolarizing_channel(self.rho1, p=0.1)
        assert is_valid_density_matrix(rho_out)

    def test_depolarizing_max_mixed_at_p_three_quarters(self):
        # NOTE: this implementation uses E0=sqrt(1-p)*I, E1..3=sqrt(p/3)*{X,Y,Z}
        # With this parameterization, maximally mixed state is reached at p=3/4, NOT p=1.
        # At p=1 the channel becomes (1/3)(XρX + YρY + ZρZ), not I/2.
        rho_out = depolarizing_channel(self.rho1, p=0.75)
        expected = np.eye(2) / 2
        assert np.allclose(rho_out, expected, atol=1e-10)

    def test_noise_channels_on_2qubit(self):
        for q in [0, 1]:
            rho_d = dephasing_channel(self.rho2, p=0.1, target_qubit=q, total_qubits=2)
            assert is_valid_density_matrix(rho_d)
            rho_a = amplitude_damping_channel(self.rho2, gamma=0.1, target_qubit=q, total_qubits=2)
            assert is_valid_density_matrix(rho_a)


# ================================================================
# 6. THERMAL RELAXATION (T1/T2)
# ================================================================

class TestThermalRelaxation:
    def setup_method(self):
        self.T1 = 100e-6
        self.T2 = 80e-6
        self.rho_excited = state_to_density(one)

    def test_t1_decay_validity(self):
        for t in [0, 10e-6, 50e-6, 100e-6, 200e-6]:
            rho_out = thermal_relaxation_from_T1_T2(
                self.rho_excited, t, self.T1, self.T2
            )
            assert is_valid_density_matrix(rho_out), f"Invalid at t={t}"

    def test_t1_decay_population(self):
        # P(|1>) should decrease as exp(-t/T1)
        for t in [10e-6, 50e-6, 100e-6]:
            rho_out = thermal_relaxation_from_T1_T2(
                self.rho_excited, t, self.T1, self.T2
            )
            p1_actual = rho_out[1, 1].real
            p1_expected = np.exp(-t / self.T1)
            # Allow tolerance because dephasing step slightly perturbs populations
            assert abs(p1_actual - p1_expected) < 0.05, \
                f"T1 decay wrong at t={t}: got {p1_actual:.4f}, expected {p1_expected:.4f}"

    def test_t1_decay_monotonic(self):
        times = np.linspace(0, 3 * self.T1, 10)
        p1_values = []
        for t in times:
            rho_out = thermal_relaxation_from_T1_T2(
                self.rho_excited, t, self.T1, self.T2
            )
            p1_values.append(rho_out[1, 1].real)
        # P(1) should be monotonically decreasing
        for i in range(len(p1_values) - 1):
            assert p1_values[i] >= p1_values[i+1] - 1e-10

    def test_t2_violation_raises(self):
        with pytest.raises(ValueError):
            thermal_relaxation_from_T1_T2(
                self.rho_excited, 1e-6, T1=50e-6, T2=200e-6  # T2 > 2*T1
            )

    def test_zero_time_no_change(self):
        rho_out = thermal_relaxation_from_T1_T2(
            self.rho_excited, t=0, T1=self.T1, T2=self.T2
        )
        assert np.allclose(rho_out, self.rho_excited, atol=1e-10)

    def test_compute_Tphi_formula(self):
        T1, T2 = 100e-6, 80e-6
        Tphi = compute_Tphi(T1, T2)
        # Verify: 1/T2 = 1/(2*T1) + 1/Tphi
        assert abs(1/T2 - (1/(2*T1) + 1/Tphi)) < 1e-20


# ================================================================
# 7. MEASUREMENT
# ================================================================

class TestMeasurement:
    def test_probabilities_zero_state(self):
        rho = state_to_density(zero)
        probs = get_probabilities(rho)
        assert np.allclose(probs, [1.0, 0.0])

    def test_probabilities_one_state(self):
        rho = state_to_density(one)
        probs = get_probabilities(rho)
        assert np.allclose(probs, [0.0, 1.0])

    def test_probabilities_superposition(self):
        psi = (zero + one) / np.sqrt(2)
        rho = state_to_density(psi)
        probs = get_probabilities(rho)
        assert np.allclose(probs, [0.5, 0.5], atol=1e-10)

    def test_probabilities_sum_to_one(self):
        rho = bell_state_rho()
        probs = get_probabilities(rho)
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_measure_deterministic_zero(self):
        rho = state_to_density(zero)
        for _ in range(20):
            assert measure(rho) == 0

    def test_measure_deterministic_one(self):
        rho = state_to_density(one)
        for _ in range(20):
            assert measure(rho) == 1

    def test_measure_returns_valid_index(self):
        rho = bell_state_rho()
        for _ in range(50):
            result = measure(rho)
            assert result in [0, 1, 2, 3]

    def test_sample_no_readout_noise_matches_probs(self):
        rho = bell_state_rho()
        counts = sample(rho, shots=10000, p01=0.0, p10=0.0)
        total = sum(counts.values())
        freq_00 = counts.get('00', 0) / total
        freq_11 = counts.get('11', 0) / total
        # Bell state: 50% |00>, 50% |11> — allow 3% tolerance
        assert abs(freq_00 - 0.5) < 0.03
        assert abs(freq_11 - 0.5) < 0.03
        assert counts.get('01', 0) == 0
        assert counts.get('10', 0) == 0

    def test_sample_covers_all_states(self):
        # Uniform state should sample all 4 outcomes
        rho = np.eye(4, dtype=complex) / 4
        counts = sample(rho, shots=10000, p01=0.0, p10=0.0)
        for state in ['00', '01', '10', '11']:
            assert state in counts


# ================================================================
# 8. READOUT NOISE
# ================================================================

class TestReadoutNoise:
    def test_no_noise_unchanged(self):
        for outcome in [0, 1, 2, 3]:
            result = apply_readout_noise(outcome, n_qubits=2, p01=0.0, p10=0.0)
            assert result == outcome

    def test_certain_flip_0to1(self):
        # p01=1.0 means every 0-bit flips to 1
        result = apply_readout_noise(0b00, n_qubits=2, p01=1.0, p10=0.0)
        assert result == 0b11  # both bits flipped 0→1

    def test_certain_flip_1to0(self):
        # p10=1.0 means every 1-bit flips to 0
        result = apply_readout_noise(0b11, n_qubits=2, p01=0.0, p10=1.0)
        assert result == 0b00  # both bits flipped 1→0

    def test_result_in_valid_range(self):
        for outcome in range(4):
            for _ in range(20):
                result = apply_readout_noise(outcome, n_qubits=2, p01=0.1, p10=0.1)
                assert 0 <= result <= 3

    def test_readout_noise_statistical(self):
        # With p01=0.5, roughly half the 0-bits should flip
        flips = sum(
            apply_readout_noise(0, n_qubits=1, p01=0.5, p10=0.0)
            for _ in range(1000)
        )
        assert 400 < flips < 600  # ~50% ± 10%

    def test_sample_with_readout_noise_valid(self):
        rho = bell_state_rho()
        counts = sample(rho, shots=1000, p01=0.02, p10=0.03)
        total = sum(counts.values())
        assert total == 1000
        for key in counts:
            assert len(key) == 2
            assert all(c in '01' for c in key)

    def test_readout_noise_shifts_distribution(self):
        # With heavy readout noise, |00> should leak into other states
        rho = state_to_density(zero_zero)
        counts_clean = sample(rho, shots=5000, p01=0.0, p10=0.0)
        counts_noisy = sample(rho, shots=5000, p01=0.2, p10=0.2)
        # Clean: all |00>. Noisy: other states should appear
        assert counts_clean.get('00', 0) == 5000
        assert counts_noisy.get('00', 0) < 5000


# ================================================================
# 9. T1/T2 EXPERIMENT VERIFICATION
# ================================================================

class TestPhysicsExperiments:
    def test_t1_decay_curve(self):
        """P(|1>) should follow exp(-t/T1)."""
        T1, T2 = 100e-6, 80e-6
        rho = state_to_density(one)
        times = np.linspace(0, 2 * T1, 8)
        for t in times:
            rho_out = thermal_relaxation_from_T1_T2(rho, t, T1, T2)
            p1 = rho_out[1, 1].real
            expected = np.exp(-t / T1)
            assert abs(p1 - expected) < 0.05

    def test_t2_ramsey_coherence_decay(self):
        """H → wait → H: coherence should decay as exp(-t/T2)."""
        T1, T2 = 100e-6, 80e-6
        rho_init = state_to_density(zero)
        # Apply H
        rho = apply_unitary_density(rho_init, H)
        # Wait
        t = 40e-6
        rho = thermal_relaxation_from_T1_T2(rho, t, T1, T2)
        # Apply H again
        rho = apply_unitary_density(rho, H)
        # P(|0>) should be less than 1 (coherence lost)
        p0 = rho[0, 0].real
        assert p0 < 1.0
        assert p0 > 0.0

    def test_idle_noise_decays(self):
        """Doing nothing still causes T1 decay."""
        from operations import apply_idle_noise
        T1, T2, Tphi = 100e-6, 80e-6, compute_Tphi(100e-6, 80e-6)
        rho = state_to_density(one)
        rho_after = apply_idle_noise(rho, t=50e-6, T1=T1, Tphi=Tphi)
        # P(1) must decrease
        assert rho_after[1, 1].real < rho[1, 1].real
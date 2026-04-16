import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def is_valid_density_matrix(rho, tol=1e-6):
    """Check trace=1, Hermitian, positive semidefinite."""
    if not np.isclose(np.trace(rho).real, 1.0, atol=tol):
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.any(eigenvalues < -tol):
        return False
    return True


# ═══════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════

class TestConfig:
    def test_T2_leq_2T1(self):
        """Physical constraint: T2 ≤ 2*T1 always."""
        from simulator.config import T1, T2
        assert T2 <= 2 * T1, f"T2={T2} > 2*T1={2*T1} — unphysical"

    def test_Tphi_positive(self):
        """Tphi must be positive."""
        from simulator.config import Tphi
        assert Tphi > 0

    def test_gate_times_positive(self):
        """All gate times must be positive."""
        from simulator.config import GATE_TIMES
        for name, t in GATE_TIMES.items():
            assert t > 0, f"Gate time for {name} is not positive"

    def test_required_gates_present(self):
        """H and CNOT must be in GATE_TIMES (used in main pipeline)."""
        from simulator.config import GATE_TIMES
        assert "H" in GATE_TIMES
        assert "CNOT" in GATE_TIMES


# ═══════════════════════════════════════════════════════
# 2. STATES
# ═══════════════════════════════════════════════════════

class TestStates:
    def test_zero_state_normalized(self):
        from simulator.states import zero
        assert np.isclose(np.linalg.norm(zero), 1.0)

    def test_one_state_normalized(self):
        from simulator.states import one
        assert np.isclose(np.linalg.norm(one), 1.0)

    def test_state_to_density_trace(self):
        """Density matrix of a pure state has trace 1."""
        from simulator.states import zero, state_to_density
        rho = state_to_density(zero)
        assert np.isclose(np.trace(rho).real, 1.0)

    def test_state_to_density_pure(self):
        """Pure state has trace(rho^2) = 1."""
        from simulator.states import zero, state_to_density
        rho = state_to_density(zero)
        assert np.isclose(np.trace(rho @ rho).real, 1.0)

    def test_zero_density_valid(self):
        from simulator.states import zero_density
        rho = zero_density(2)
        assert is_valid_density_matrix(rho)

    def test_zero_density_correct_size(self):
        from simulator.states import zero_density
        rho = zero_density(3)
        assert rho.shape == (8, 8)


# ═══════════════════════════════════════════════════════
# 3. GATES
# ═══════════════════════════════════════════════════════

class TestGates:
    def test_H_unitary(self):
        from simulator.gates import H
        assert np.allclose(H @ H.conj().T, np.eye(2))

    def test_X_unitary(self):
        from simulator.gates import X
        assert np.allclose(X @ X.conj().T, np.eye(2))

    def test_CNOT_unitary(self):
        from simulator.gates import CNOT
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4))

    def test_H_squared_is_identity(self):
        """H is its own inverse."""
        from simulator.gates import H
        assert np.allclose(H @ H, np.eye(2), atol=1e-10)

    def test_X_flips_zero_to_one(self):
        from simulator.gates import X
        zero = np.array([1, 0], dtype=complex)
        one  = np.array([0, 1], dtype=complex)
        assert np.allclose(X @ zero, one)

    def test_CNOT_flips_target_when_control_is_1(self):
        """CNOT |10> → |11>"""
        from simulator.gates import CNOT
        state_10 = np.array([0, 0, 1, 0], dtype=complex)
        state_11 = np.array([0, 0, 0, 1], dtype=complex)
        assert np.allclose(CNOT @ state_10, state_11)

    def test_CNOT_does_nothing_when_control_is_0(self):
        """CNOT |00> → |00>"""
        from simulator.gates import CNOT
        state_00 = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(CNOT @ state_00, state_00)


# ═══════════════════════════════════════════════════════
# 4. EXPAND
# ═══════════════════════════════════════════════════════

class TestExpand:
    def test_expand_on_qubit0_correct_size(self):
        from simulator.expand import expand_single_qubit_gate
        from simulator.gates import H
        G = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
        assert G.shape == (4, 4)

    def test_expand_on_qubit1_correct_size(self):
        from simulator.expand import expand_single_qubit_gate
        from simulator.gates import H
        G = expand_single_qubit_gate(H, target_qubit=1, total_qubits=2)
        assert G.shape == (4, 4)

    def test_expanded_gate_is_unitary(self):
        from simulator.expand import expand_single_qubit_gate
        from simulator.gates import H
        G = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
        assert np.allclose(G @ G.conj().T, np.eye(4), atol=1e-10)

    def test_kraus_expansion_preserves_count(self):
        from simulator.expand import expand_kraus_to_n_qubits
        from simulator.noise import amplitude_damping_kraus
        kraus = amplitude_damping_kraus(0.1)
        expanded = expand_kraus_to_n_qubits(kraus, target_qubit=0, total_qubits=2)
        assert len(expanded) == len(kraus)

    def test_kraus_expansion_correct_size(self):
        from simulator.expand import expand_kraus_to_n_qubits
        from simulator.noise import amplitude_damping_kraus
        kraus = amplitude_damping_kraus(0.1)
        expanded = expand_kraus_to_n_qubits(kraus, target_qubit=0, total_qubits=2)
        for K in expanded:
            assert K.shape == (4, 4)


# ═══════════════════════════════════════════════════════
# 5. LINALG
# ═══════════════════════════════════════════════════════

class TestLinalg:
    def test_normalize_unit_vector(self):
        from simulator.linalg import normalize
        v = np.array([3, 4], dtype=complex)
        n = normalize(v)
        assert np.isclose(np.linalg.norm(n), 1.0)

    def test_check_kraus_valid(self):
        """Amplitude damping Kraus ops should satisfy completeness."""
        from simulator.linalg import check_kraus
        from simulator.noise import amplitude_damping_kraus
        kraus = amplitude_damping_kraus(0.1)
        assert check_kraus(kraus)

    def test_check_kraus_dephasing(self):
        from simulator.linalg import check_kraus
        from simulator.noise import dephasing_kraus
        kraus = dephasing_kraus(0.1)
        assert check_kraus(kraus)

    def test_check_kraus_depolarizing(self):
        from simulator.linalg import check_kraus
        from simulator.noise import depolarizing_kraus
        kraus = depolarizing_kraus(0.1)
        assert check_kraus(kraus)


# ═══════════════════════════════════════════════════════
# 6. NOISE CHANNELS
# ═══════════════════════════════════════════════════════

class TestNoiseChannels:

    def _bell_state_rho(self):
        """Helper: returns clean Bell state density matrix."""
        from simulator.states import state_to_density
        from simulator.gates import H, CNOT
        from simulator.expand import expand_single_qubit_gate
        from simulator.operations import apply_unitary_density
        psi = np.array([1, 0, 0, 0], dtype=complex)
        H0 = expand_single_qubit_gate(H, 0, 2)
        psi = H0 @ psi
        psi = CNOT @ psi
        psi = psi / np.linalg.norm(psi)
        return state_to_density(psi)

    def test_dephasing_preserves_trace(self):
        from simulator.noise import dephasing_channel
        rho = self._bell_state_rho()
        rho_out = dephasing_channel(rho, p=0.1, target_qubit=0, total_qubits=2)
        assert np.isclose(np.trace(rho_out).real, 1.0, atol=1e-6)

    def test_amplitude_damping_preserves_trace(self):
        from simulator.noise import amplitude_damping_channel
        rho = self._bell_state_rho()
        rho_out = amplitude_damping_channel(rho, gamma=0.1, target_qubit=0, total_qubits=2)
        assert np.isclose(np.trace(rho_out).real, 1.0, atol=1e-6)

    def test_depolarizing_preserves_trace(self):
        from simulator.noise import depolarizing_channel
        rho = self._bell_state_rho()
        rho_out = depolarizing_channel(rho, p=0.1, target_qubit=0, total_qubits=2)
        assert np.isclose(np.trace(rho_out).real, 1.0, atol=1e-6)

    def test_dephasing_output_is_valid_density_matrix(self):
        from simulator.noise import dephasing_channel
        rho = self._bell_state_rho()
        rho_out = dephasing_channel(rho, p=0.1, target_qubit=0, total_qubits=2)
        assert is_valid_density_matrix(rho_out)

    def test_amplitude_damping_output_is_valid_density_matrix(self):
        from simulator.noise import amplitude_damping_channel
        rho = self._bell_state_rho()
        rho_out = amplitude_damping_channel(rho, gamma=0.1, target_qubit=0, total_qubits=2)
        assert is_valid_density_matrix(rho_out)

    def test_zero_noise_is_identity(self):
        """p=0 dephasing should leave state unchanged."""
        from simulator.noise import dephasing_channel
        rho = self._bell_state_rho()
        rho_out = dephasing_channel(rho, p=0.0, target_qubit=0, total_qubits=2)
        assert np.allclose(rho_out, rho, atol=1e-10)

    def test_dephasing_kills_coherence(self):
        """p=0.5 dephasing should reduce off-diagonal elements."""
        from simulator.noise import dephasing_channel
        rho = self._bell_state_rho()
        rho_out = dephasing_channel(rho, p=0.5, target_qubit=0, total_qubits=2)
        # Off-diagonal coherence should be strictly smaller
        assert abs(rho_out[0, 3]) < abs(rho[0, 3])

    def test_apply_noise_uses_config_timing(self):
        """apply_noise should run without error using config GATE_TIMES."""
        from simulator.noise import apply_noise
        rho = self._bell_state_rho()
        rho_out = apply_noise(rho, "H", [0], total_qubits=2)
        assert is_valid_density_matrix(rho_out)

    def test_apply_noise_cnot_both_qubits(self):
        from simulator.noise import apply_noise
        rho = self._bell_state_rho()
        rho_out = apply_noise(rho, "CNOT", [0, 1], total_qubits=2)
        assert is_valid_density_matrix(rho_out)

    def test_thermal_relaxation_valid(self):
        from simulator.noise import thermal_relaxation_channel
        from simulator.config import T1, Tphi
        rho = self._bell_state_rho()
        rho_out = thermal_relaxation_channel(rho, t=100e-9, T1=T1, Tphi=Tphi,
                                              target_qubit=0, total_qubits=2)
        assert is_valid_density_matrix(rho_out)


# ═══════════════════════════════════════════════════════
# 7. OPERATIONS
# ═══════════════════════════════════════════════════════

class TestOperations:
    def test_apply_unitary_preserves_trace(self):
        from simulator.operations import apply_unitary_density
        from simulator.states import zero_density
        from simulator.gates import H
        from simulator.expand import expand_single_qubit_gate
        rho = zero_density(2)
        H0 = expand_single_qubit_gate(H, 0, 2)
        rho_out = apply_unitary_density(rho, H0)
        assert np.isclose(np.trace(rho_out).real, 1.0)

    def test_apply_idle_noise_valid(self):
        from simulator.operations import apply_idle_noise
        from simulator.states import zero_density
        rho = zero_density(2)
        rho_out = apply_idle_noise(rho, idle_time=100e-9, target_qubit=1, total_qubits=2)
        assert is_valid_density_matrix(rho_out)


# ═══════════════════════════════════════════════════════
# 8. CIRCUIT + FULL PIPELINE
# ═══════════════════════════════════════════════════════

class TestCircuit:
    def test_gateop_stores_fields(self):
        from simulator.circuit import GateOp
        from simulator.gates import H
        from simulator.expand import expand_single_qubit_gate
        H0 = expand_single_qubit_gate(H, 0, 2)
        op = GateOp(name="H", matrix=H0, targets=[0])
        assert op.name == "H"
        assert op.targets == [0]
        assert op.matrix.shape == (4, 4)

    def test_full_pipeline_trace(self):
        """Bell state pipeline: trace should stay ~1.0 throughout."""
        from simulator.circuit import GateOp
        from simulator.gates import H, CNOT
        from simulator.expand import expand_single_qubit_gate
        from simulator.operations import apply_unitary_density, apply_idle_noise
        from simulator.noise import apply_noise
        from simulator.states import state_to_density
        from simulator.config import GATE_TIMES

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = state_to_density(psi)

        H0 = expand_single_qubit_gate(H, 0, 2)
        circuit = [
            GateOp("H",    H0,   [0]),
            GateOp("CNOT", CNOT, [0, 1]),
        ]

        total_qubits = 2
        for op in circuit:
            rho = apply_unitary_density(rho, op.matrix)
            rho = apply_noise(rho, op.name, op.targets, total_qubits)
            idle = [q for q in range(total_qubits) if q not in op.targets]
            for q in idle:
                rho = apply_idle_noise(rho, GATE_TIMES[op.name], q, total_qubits)

        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)

    def test_full_pipeline_valid_density_matrix(self):
        """Final state after noisy pipeline must be a valid density matrix."""
        from simulator.circuit import GateOp
        from simulator.gates import H, CNOT
        from simulator.expand import expand_single_qubit_gate
        from simulator.operations import apply_unitary_density, apply_idle_noise
        from simulator.noise import apply_noise
        from simulator.states import state_to_density
        from simulator.config import GATE_TIMES

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = state_to_density(psi)
        H0 = expand_single_qubit_gate(H, 0, 2)
        circuit = [
            GateOp("H",    H0,   [0]),
            GateOp("CNOT", CNOT, [0, 1]),
        ]
        total_qubits = 2
        for op in circuit:
            rho = apply_unitary_density(rho, op.matrix)
            rho = apply_noise(rho, op.name, op.targets, total_qubits)
            idle = [q for q in range(total_qubits) if q not in op.targets]
            for q in idle:
                rho = apply_idle_noise(rho, GATE_TIMES[op.name], q, total_qubits)

        assert is_valid_density_matrix(rho)

    def test_noise_spreads_distribution(self):
        """Noisy Bell state should have non-zero |01> and |10> probabilities."""
        from simulator.circuit import GateOp
        from simulator.gates import H, CNOT
        from simulator.expand import expand_single_qubit_gate
        from simulator.operations import apply_unitary_density, apply_idle_noise
        from simulator.noise import apply_noise
        from simulator.states import state_to_density
        from simulator.config import GATE_TIMES

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = state_to_density(psi)
        H0 = expand_single_qubit_gate(H, 0, 2)
        circuit = [
            GateOp("H",    H0,   [0]),
            GateOp("CNOT", CNOT, [0, 1]),
        ]
        total_qubits = 2
        for op in circuit:
            rho = apply_unitary_density(rho, op.matrix)
            rho = apply_noise(rho, op.name, op.targets, total_qubits)
            idle = [q for q in range(total_qubits) if q not in op.targets]
            for q in idle:
                rho = apply_idle_noise(rho, GATE_TIMES[op.name], q, total_qubits)

        # |01> or |10> must have some weight — noise broke the clean Bell state
        assert rho[1, 1].real > 1e-8 or rho[2, 2].real > 1e-8


# ═══════════════════════════════════════════════════════
# 9. MEASUREMENT
# ═══════════════════════════════════════════════════════

class TestMeasurement:
    def _noisy_bell_rho(self):
        from simulator.circuit import GateOp
        from simulator.gates import H, CNOT
        from simulator.expand import expand_single_qubit_gate
        from simulator.operations import apply_unitary_density, apply_idle_noise
        from simulator.noise import apply_noise
        from simulator.states import state_to_density
        from simulator.config import GATE_TIMES
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = state_to_density(psi)
        H0 = expand_single_qubit_gate(H, 0, 2)
        circuit = [GateOp("H", H0, [0]), GateOp("CNOT", CNOT, [0, 1])]
        for op in circuit:
            rho = apply_unitary_density(rho, op.matrix)
            rho = apply_noise(rho, op.name, op.targets, 2)
            for q in [q for q in range(2) if q not in op.targets]:
                rho = apply_idle_noise(rho, GATE_TIMES[op.name], q, 2)
        return rho

    def test_probabilities_sum_to_one(self):
        from simulator.measurement import get_probabilities
        rho = self._noisy_bell_rho()
        probs = get_probabilities(rho)
        assert np.isclose(sum(probs), 1.0, atol=1e-6)

    def test_probabilities_non_negative(self):
        from simulator.measurement import get_probabilities
        rho = self._noisy_bell_rho()
        probs = get_probabilities(rho)
        assert all(p >= -1e-10 for p in probs)

    def test_sample_counts_add_up(self):
        from simulator.measurement import sample
        rho = self._noisy_bell_rho()
        counts = sample(rho, shots=500)
        assert sum(counts.values()) == 500

    def test_sample_with_readout_noise_counts_add_up(self):
        from simulator.measurement import sample
        rho = self._noisy_bell_rho()
        counts = sample(rho, shots=500, p01=0.02, p10=0.03)
        assert sum(counts.values()) == 500

    def test_sample_keys_are_binary_strings(self):
        from simulator.measurement import sample
        rho = self._noisy_bell_rho()
        counts = sample(rho, shots=100)
        for key in counts:
            assert all(c in "01" for c in key), f"Non-binary key: {key}"
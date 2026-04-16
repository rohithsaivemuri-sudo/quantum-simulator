import numpy as np
from expand import expand_single_qubit_gate, expand_kraus_to_n_qubits

# ============================================================
# BASIC RELATIONS
# ============================================================

def compute_Tphi(T1, T2):
    denom = (1 / T2) - (1 / (2 * T1))
    if denom <= 0:
        raise ValueError("Invalid T1/T2 relation")
    return 1 / denom


# ============================================================
# KRAUS ENGINE
# ============================================================

def apply_kraus(rho, kraus_ops):
    rho_new = np.zeros_like(rho, dtype=complex)
    for E in kraus_ops:
        rho_new += E @ rho @ E.conj().T
    return rho_new


# ============================================================
# AMPLITUDE DAMPING (T1)
# ============================================================

def amplitude_damping_kraus(gamma):
    K0 = np.array([
        [1, 0],
        [0, np.sqrt(1 - gamma)]
    ], dtype=complex)

    K1 = np.array([
        [0, np.sqrt(gamma)],
        [0, 0]
    ], dtype=complex)

    return [K0, K1]


def amplitude_damping_channel(rho, gamma, target_qubit=0, total_qubits=1):
    kraus = amplitude_damping_kraus(gamma)
    kraus_full = expand_kraus_to_n_qubits(kraus, target_qubit, total_qubits)
    return apply_kraus(rho, kraus_full)


# ============================================================
# DEPHASING (Tphi) — CORRECT MAPPING
# ============================================================

def dephasing_kraus(p, target_qubit=0, total_qubits=1):
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z

    if total_qubits == 1:
        return [E0, E1]

    E0_full = expand_single_qubit_gate(E0, target_qubit, total_qubits)
    E1_full = expand_single_qubit_gate(E1, target_qubit, total_qubits)

    return [E0_full, E1_full]


def dephasing_channel(rho, p, target_qubit=0, total_qubits=1):
    return apply_kraus(rho, dephasing_kraus(p, target_qubit, total_qubits))


# ============================================================
# THERMAL RELAXATION (T1 + Tphi)
# ============================================================

def thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit=0, total_qubits=1):

    # amplitude damping probability
    gamma = 1 - np.exp(-t / T1)

    # ✅ CORRECT dephasing probability
    p_phi = (1 - np.exp(-t / Tphi)) / 2

    rho = amplitude_damping_channel(rho, gamma, target_qubit, total_qubits)
    rho = dephasing_channel(rho, p_phi, target_qubit, total_qubits)

    return rho


def apply_global_thermal_noise(rho, t, T1, Tphi, total_qubits):

    for q in range(total_qubits):
        rho = thermal_relaxation_channel(
            rho, t, T1, Tphi,
            target_qubit=q,
            total_qubits=total_qubits
        )

    return rho


# ============================================================
# PURE DEPHASING (FOR RAMSEY)
# ============================================================

def pure_dephasing_global(rho, t, Tphi, total_qubits):

    # ✅ SAME correct mapping
    p_phi = (1 - np.exp(-t / Tphi)) / 2

    for q in range(total_qubits):
        rho = dephasing_channel(rho, p_phi, q, total_qubits)

    return rho


# ============================================================
# OPTIONAL GATE NOISE
# ============================================================

from config import T1, Tphi, GATE_TIMES

def apply_noise(rho, gate_name, target_qubits, total_qubits):

    t = GATE_TIMES[gate_name]

    gamma = 1 - np.exp(-t / T1)
    p_phi = (1 - np.exp(-t / Tphi)) / 2

    for q in target_qubits:
        rho = amplitude_damping_channel(rho, gamma, q, total_qubits)
        rho = dephasing_channel(rho, p_phi, q, total_qubits)

    return rho


# ============================================================
# UTIL
# ============================================================

def normalize_density_matrix(rho):
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho
import numpy as np
from simulator.expand import expand_single_qubit_gate, expand_kraus_to_n_qubits
from config import GATE_TIMES, T1, T2, Tphi

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


def depolarizing_kraus(p):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return [
        np.sqrt(1 - p) * I,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def depolarizing_channel(rho, p, target_qubit=0, total_qubits=1):
    kraus = depolarizing_kraus(p)
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


def thermal_relaxation_from_T1_T2(rho, t, T1, T2, target_qubit=0, total_qubits=1):
    Tphi = compute_Tphi(T1, T2)
    return thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit, total_qubits)


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

def apply_noise(rho, dt, target_qubits=None, total_qubits=2):
    if isinstance(dt, str):
        dt = GATE_TIMES[dt]

    gamma = 1 - np.exp(-dt / T1) if T1 > 0 else 0.0
    lam = 1 - np.exp(-dt / Tphi) if Tphi > 0 else 0.0
    p_phi = lam / 2

    qubits = range(total_qubits) if target_qubits is None else target_qubits
    for q in qubits:
        rho = amplitude_damping_channel(rho, gamma, q, total_qubits)
        rho = dephasing_channel(rho, p_phi, q, total_qubits)

    return normalize_density_matrix(rho)


# ============================================================
# UTIL
# ============================================================

def normalize_density_matrix(rho):
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho

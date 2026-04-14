import numpy as np
from expand import expand_single_qubit_gate, expand_kraus_to_n_qubits

def get_pa(t, T1):
    return 1 - np.exp(-t / T1)

def get_pp(t, Tphi):
    return 1 - np.exp(-t / Tphi)

def compute_Tphi(T1, T2):
    return 1 / (1/T2 - 1/(2*T1))

def thermal_relaxation_from_T1_T2(rho, t, T1, T2, target_qubit=0, total_qubits=1):
    if T2 > 2 * T1:
        raise ValueError(f"T2 ({T2}) cannot exceed 2*T1 ({T1})")
    Tphi = compute_Tphi(T1, T2)
    return thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit, total_qubits)

def apply_kraus(rho, kraus_ops):
    rho_new = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        rho_new += E @ rho @ E.conj().T

    return rho_new


# ------------------ DEPHASING ------------------

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


# ------------------ AMPLITUDE DAMPING ------------------

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


# ------------------ DEPOLARIZING ------------------

def depolarizing_kraus(p, target_qubit=0, total_qubits=1):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p/3) * X
    E2 = np.sqrt(p/3) * Y
    E3 = np.sqrt(p/3) * Z

    if total_qubits == 1:
        return [E0, E1, E2, E3]

    E0_full = expand_single_qubit_gate(E0, target_qubit, total_qubits)
    E1_full = expand_single_qubit_gate(E1, target_qubit, total_qubits)
    E2_full = expand_single_qubit_gate(E2, target_qubit, total_qubits)
    E3_full = expand_single_qubit_gate(E3, target_qubit, total_qubits)

    return [E0_full, E1_full, E2_full, E3_full]


def depolarizing_channel(rho, p, target_qubit=0, total_qubits=1):
    return apply_kraus(rho, depolarizing_kraus(p, target_qubit, total_qubits))


# ------------------ NOISE PIPELINE ------------------

# ADD THIS to noise.py
from config import T1, Tphi, GATE_TIMES

def apply_noise(rho, gate_name, target_qubits, total_qubits):
    """
    Looks up gate duration from config, computes noise params,
    applies thermal relaxation to every target qubit.
    """
    t = GATE_TIMES[gate_name]

    gamma = 1 - np.exp(-t / T1)          # amplitude damping strength
    p_phi = 1 - np.exp(-t / Tphi)        # dephasing strength
    p_phi = min(p_phi, 0.5)              # physical cap

    for q in target_qubits:
        rho = amplitude_damping_channel(rho, gamma, q, total_qubits)
        rho = dephasing_channel(rho, p_phi, q, total_qubits)

    return rho

# ------------------ THERMAL RELAXATION (TIME-DEPENDENT) ------------------




def thermal_relaxation_channel(rho, t, T1, Tphi, target_qubit=0, total_qubits=1):
    
    p_a = get_pa(t, T1)      # amplitude damping
    p_p = min(get_pp(t, Tphi), 0.5)
    # Apply amplitude damping
    rho = amplitude_damping_channel(rho, p_a, target_qubit, total_qubits)

    # Apply dephasing
    rho = dephasing_channel(rho, p_p, target_qubit, total_qubits)

    return rho

def apply_global_thermal_noise(rho, t, T1, Tphi, total_qubits):
    
    for q in range(total_qubits):
        rho = thermal_relaxation_channel(
            rho, t, T1, Tphi,
            target_qubit=q,
            total_qubits=total_qubits
        )
    
    return rho

def normalize_density_matrix(rho):
    # Make Hermitian
    rho = (rho + rho.conj().T) / 2
    
    # Normalize trace
    rho = rho / np.trace(rho)
    
    return rho




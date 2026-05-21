# exp5_echo_revival.py
import numpy as np
from simulator.states import state_to_density
from simulator.operations import apply_unitary_density
from simulator.noise import apply_global_thermal_noise, compute_Tphi
from simulator.gates import H, X, Z, I
from simulator.config import T1, T2

def apply_z_rotation(rho, theta):
    """Applies a coherent Z-rotation (detuning phase)."""
    Rz = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * Z
    return apply_unitary_density(rho, Rz)

def run_ensemble_echo():
    tau_times = [0, 5e-6, 10e-6, 25e-6, 50e-6, 100e-6]
    
    # 1. Create an ensemble of detuning frequencies (Gaussian distribution)
    # This represents the inhomogeneous magnetic field causing T2*
    np.random.seed(42)
    ensemble_size = 500
    # Standard deviation of 100 kHz detuning
    detuning_frequencies = np.random.normal(loc=0.0, scale=100e3 * (2 * np.pi), size=ensemble_size)
    
    Tphi = compute_Tphi(T1, T2)
    psi0 = np.array([1, 0], dtype=complex)

    print("\n--- T2* vs T2 (Ensemble Detuning) ---")
    print(f"{'Wait Time (2*tau)':<20} | {'Ramsey P(0) (T2*)':<20} | {'Echo P(0) (T2)':<20}")
    print("-" * 65)

    for tau in tau_times:
        total_time = 2 * tau
        
        # Accumulators for the ensemble average
        ramsey_p0_sum = 0.0
        echo_p0_sum = 0.0

        for delta_omega in detuning_frequencies:
            # The phase accumulated during time tau
            theta = delta_omega * tau 

            # --- RAMSEY SEQUENCE (No Echo) ---
            rho_ramsey = state_to_density(psi0)
            rho_ramsey = apply_unitary_density(rho_ramsey, H)
            
            # Wait 2*tau (accumulate double the phase and Markovian noise)
            rho_ramsey = apply_z_rotation(rho_ramsey, theta * 2)
            rho_ramsey = apply_global_thermal_noise(rho_ramsey, t=total_time, T1=T1, Tphi=Tphi, total_qubits=1)
            
            rho_ramsey = apply_unitary_density(rho_ramsey, H)
            ramsey_p0_sum += np.real(rho_ramsey[0, 0])

            # --- HAHN ECHO SEQUENCE ---
            rho_echo = state_to_density(psi0)
            rho_echo = apply_unitary_density(rho_echo, H)
            
            # First wait (tau): accumulate phase + noise
            rho_echo = apply_z_rotation(rho_echo, theta)
            rho_echo = apply_global_thermal_noise(rho_echo, t=tau, T1=T1, Tphi=Tphi, total_qubits=1)
            
            # The PI-PULSE! (Flips the phase)
            rho_echo = apply_unitary_density(rho_echo, X)
            
            # Second wait (tau): accumulate more phase (which unwinds the first phase!) + noise
            rho_echo = apply_z_rotation(rho_echo, theta)
            rho_echo = apply_global_thermal_noise(rho_echo, t=tau, T1=T1, Tphi=Tphi, total_qubits=1)
            
            rho_echo = apply_unitary_density(rho_echo, H)
            echo_p0_sum += np.real(rho_echo[0, 0])

        # Calculate ensemble averages
        ramsey_avg = ramsey_p0_sum / ensemble_size
        echo_avg = echo_p0_sum / ensemble_size

        print(f"{total_time:<20.1e} | {ramsey_avg:<20.4f} | {echo_avg:<20.4f}")

if __name__ == "__main__":
    run_ensemble_echo()
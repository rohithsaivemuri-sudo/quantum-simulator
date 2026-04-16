 
# main.py
import numpy as np
from circuit import GateOp
from expand import expand_single_qubit_gate
from operations import apply_unitary_density
from noise import apply_noise, apply_global_thermal_noise, compute_Tphi
from states import state_to_density
from gates import H, CNOT
from measurement import get_probabilities, sample
from config import GATE_TIMES, T1, T2
 
total_qubits = 2
 
 
def normalize(state):
    return state / np.linalg.norm(state)
 
 
def print_analysis(rho, label=""):
    print(f"\n--- {label} ---")
    print(f"Trace:    {np.trace(rho).real:.6f}")
    print(f"P(|00>) = {rho[0,0].real:.6f}")
    print(f"P(|01>) = {rho[1,1].real:.6f}")
    print(f"P(|10>) = {rho[2,2].real:.6f}")
    print(f"P(|11>) = {rho[3,3].real:.6f}")
    print(f"Coherence rho[0,3] = {rho[0,3]}")
 
 
def run_circuit(circuit, rho, total_qubits):
    for op in circuit:
 
        # ------------------ WAIT (TIME EVOLUTION) ------------------
        if op.name == "WAIT":
            # BUG FIX: Previously used GATE_TIMES.get("WAIT", 1.0).
            # The default of 1.0 second is 50,000× longer than T1 (20µs),
            # which completely decays the qubit to |0> — a physically catastrophic error.
            # WAIT is now a proper key in GATE_TIMES (1µs default). Use [] not .get().
            t = GATE_TIMES["WAIT"]
            Tphi = compute_Tphi(T1, T2)
 
            rho = apply_global_thermal_noise(
                rho, t, T1, Tphi, total_qubits
            )
 
            print_analysis(rho, f"After WAIT (t={t})")
            continue
 
        # ------------------ APPLY GATE ------------------
        rho = apply_unitary_density(rho, op.matrix)
        print_analysis(rho, f"After gate: {op.name} on qubits {op.targets}")
 
        # ------------------ APPLY GATE NOISE ------------------
        rho = apply_noise(rho, op.name, op.targets, total_qubits)
        print_analysis(rho, f"After noise: {op.name}")
 
        # ------------------ (OPTIONAL) IDLE NOISE ------------------
        # Disabled to avoid double-counting with WAIT
        # idle_qubits = [q for q in range(total_qubits) if q not in op.targets]
        # gate_time = GATE_TIMES[op.name]
        # for q in idle_qubits:
        #     rho = apply_idle_noise(rho, gate_time, q, total_qubits)
 
    return rho
 
 
def main():
    # Initial state |00⟩
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = state_to_density(psi)
    print_analysis(rho, "Initial |00>")
 
    # Build circuit: H → CNOT → WAIT
    H_expanded = expand_single_qubit_gate(H, target_qubit=0, total_qubits=2)
 
    circuit = [
        GateOp(name="H",    matrix=H_expanded, targets=[0]),
        GateOp(name="CNOT", matrix=CNOT,       targets=[0, 1]),
        GateOp(name="WAIT", matrix=None,       targets=[0, 1]),
    ]
 
    # Run circuit
    rho = run_circuit(circuit, rho, total_qubits)
 
    print("\n=== FINAL STATE ===")
    print(f"Trace: {np.trace(rho).real:.6f}  (should be ~1.0)")
 
    # Measurement
    probs = get_probabilities(rho)
    print("\nProbabilities:")
    for i, p in enumerate(probs):
        print(f"  |{format(i, '02b')}> : {p:.4f}")
 
    print("\n1000 shots (no readout noise):")
    counts = sample(rho, shots=1000, p01=0.0, p10=0.0)
    for state, count in sorted(counts.items()):
        print(f"  |{state}> : {count} ({count/10:.1f}%)")
 
    print("\n1000 shots (with readout noise):")
    counts_noisy = sample(rho, shots=1000, p01=0.02, p10=0.03)
    for state, count in sorted(counts_noisy.items()):
        print(f"  |{state}> : {count} ({count/10:.1f}%)")
 
 
if __name__ == "__main__":
    main()
 
from simulator.engine import Engine
from simulator.measurement import get_probabilities, sample 
from metrics.fidelity import fidelity
from simulator.states import initial_state

eng= Engine(total_qubits=2, noise_enabled=True)
eng.h(0)
eng.cnot(0,1)
eng.wait()

rho_noisy= eng.state()

rho_ideal= Engine(total_qubits=2, noise_enabled=False)
rho_ideal.h(0)
rho_ideal.cnot(0,1)
rho_ideal.wait()
rho_ideal= rho_ideal.state()

probs = get_probabilities(rho_noisy)
print("P(|00>):", probs[0])
print("P(|01>):", probs[1])
print("P(|10>):", probs[2])
print("P(|11>):", probs[3])

# Sample 1000 shots
counts = sample(rho_noisy, shots=1000, p01=0.02, p10=0.03)
print(counts)

print("Fidelity:", fidelity(rho_ideal, rho_noisy))
print("Total time elapsed:", eng.time * 1e6, "µs")
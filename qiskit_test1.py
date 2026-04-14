from qiskit import QuantumCircuit
print("Qiskit working")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator()
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()

print(counts)


from qiskit_aer.noise import NoiseModel, depolarizing_error

noise_model = NoiseModel()

error_1q = depolarizing_error(0.01, 1)
error_2q = depolarizing_error(0.05, 2)

noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])


sim = AerSimulator(noise_model=noise_model)

result = sim.run(qc, shots=1000).result()
counts_noisy = result.get_counts()

print(counts_noisy)
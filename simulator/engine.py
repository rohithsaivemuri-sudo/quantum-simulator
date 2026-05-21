from simulator.states import initial_state
from simulator.operations import apply_h, apply_cnot, apply_x, apply_y, apply_z
from simulator.noise import apply_noise
from simulator.config import GATE_TIMES

class Engine:
    def __init__(self, total_qubits=2, noise_enabled=True):
        self.total_qubits = total_qubits
        self.noise_enabled = noise_enabled
        self.reset()

    def reset(self):
        self.rho = initial_state(self.total_qubits)
        self.time = 0.0

    def _apply_gate_and_noise(self, gate_name, gate_fn, *args, target_qubits=None):
        dt = GATE_TIMES[gate_name]
        self.rho = gate_fn(self.rho, *args, total_qubits=self.total_qubits)
        if self.noise_enabled:
            self.rho = apply_noise(
                self.rho,
                dt,
                target_qubits=target_qubits,
                total_qubits=self.total_qubits,
            )
        self.time += dt
        return self.rho

    def h(self, q):
        self._apply_gate_and_noise("H", apply_h, q, target_qubits=[q])

    def x(self, q):
        self._apply_gate_and_noise("X", apply_x, q, target_qubits=[q])

    def y(self, q):
        self._apply_gate_and_noise("Y", apply_y, q, target_qubits=[q])

    def z(self, q):
        self._apply_gate_and_noise("Z", apply_z, q, target_qubits=[q])

    def cnot(self, q1, q2):
        self._apply_gate_and_noise("CNOT", apply_cnot, q1, q2, target_qubits=[q1, q2])

    def wait(self, duration=None):
        if duration is None:
            duration = GATE_TIMES["WAIT"]
        if self.noise_enabled:
            self.rho = apply_noise(self.rho, duration, total_qubits=self.total_qubits)
        self.time += duration
        return self.rho

    def state(self):
        return self.rho

from simulator.states import initial_state
from simulator.operations import apply_h, apply_cnot, apply_x, apply_y, apply_z
from simulator.noise import apply_noise

class Engine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rho = initial_state()

    def h(self, q):
        self.rho = apply_h(self.rho, q)
        self.rho = apply_noise(self.rho)

    def x(self, q):
        self.rho = apply_x(self.rho, q)
        self.rho = apply_noise(self.rho)

    def y(self, q):
        self.rho = apply_y(self.rho, q)
        self.rho = apply_noise(self.rho)

    def z(self, q):
        self.rho = apply_z(self.rho, q)
        self.rho = apply_noise(self.rho)

    def cnot(self, q1, q2):
        self.rho = apply_cnot(self.rho, q1, q2)
        self.rho = apply_noise(self.rho)

    def state(self):
        return self.rho
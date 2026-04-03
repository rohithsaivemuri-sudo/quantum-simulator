import numpy as np
from expand import expand_single_qubit_gate
from operations import apply_gate, apply_cnot
from states import state_to_density


def test_density_matrix():
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    rho = state_to_density(state)

    print("\nTest Density Matrix:")
    print(rho)

    print("Trace:", np.trace(rho))
    print("PASS:", np.isclose(np.trace(rho), 1))


if __name__ == "__main__":
    test_density_matrix()

    
import numpy as np
from expand import expand_single_qubit_gate
from operations import apply_gate, apply_cnot
from states import state_to_density

H = (1 / np.sqrt(2)) * np.array([[1, 1],[1, -1]], dtype=complex)

def test_bell_state():
    state = np.array([1, 0, 0, 0], dtype=complex)

    H0 = expand_single_qubit_gate(H, 0)
    state = apply_gate(state, H0)
    state = apply_cnot(state)

    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)

    print("\nTest Bell State:")
    print("Output:", state)
    print("Expected:", expected)

    print("PASS:", np.allclose(state, expected))


if __name__ == "__main__":
    test_bell_state()
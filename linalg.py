import numpy as np
def normalize(state):
    norm=np.sqrt(np.sum(np.abs(state)**2))
    if(norm==0):
        print("invalid states")
        return
    else:
        return state/norm
    
def tensor(a,b):
    return np.kron(a,b)


def check_kraus(kraus_ops):
    dim = kraus_ops[0].shape[0]
    identity = np.eye(dim)

    total = np.zeros((dim, dim), dtype=complex)

    for E in kraus_ops:
        total += E.conj().T @ E

    return np.allclose(total, identity)
import numpy as np


class pauli:
    @property
    def i(self):
        return np.eyes(2, dtype=np.complex128)

    @property
    def x(self):
        return np.array([
            [0., 1.],
            [1., 0.]
        ], dtype=np.complex128)

    @property
    def y(self):
        return np.array([
            [0., -1.j],
            [1.j, 0.]
        ], dtype=np.complex128)

    @property
    def z(self):
        return np.array([
            [1., 0.],
            [0., -1.]
        ], dtype=np.complex128)


σ = pauli()


class dirac_impl:
    def __init__(self, N) -> None:
        self.N = N

    def bra(self, m: int):
        return self.ket(m).T.conj()

    def ket(self, m):
        state = torch.tensor([[0.]] * self.N, dtype=torch.cdouble)
        state[m][0] = 1.
        return state


class dirac:
    def __init__(self, *Ns) -> None:
        self.ds = []
        for N in Ns:
            self.ds.append(dirac_impl(N))

    def bra(self, *ms):
        return self.ket(*ms).T.conj()

    def ket(self, *ms):
        state = torch.tensor([[1.]], dtype=torch.cdouble)
        for i, m in enumerate(ms):
            state = torch.kron(state, self.ds[i].ket(m))
        return state
